import base64
import time
from typing import Any

import requests
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterList,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ConnectTimeout, Timeout

SERVICE = "BlackForest Labs"
API_KEY_ENV_VAR = "BFL_API_KEY"
BFL_API_BASE_URL = "https://api.bfl.ai"

# Constants
MAX_INPUT_IMAGES = 8  # API supports up to input_image_8
ASPECT_RATIO_OPTIONS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21", "3:7", "7:3"]
OUTPUT_FORMAT_OPTIONS = ["jpeg", "png"]
MODEL_OPTIONS = ["flux-2-pro", "flux-2-flex"]
SAFETY_TOLERANCE_OPTIONS = ["least restrictive", "moderate", "most restrictive"]


class Flux2ImageGeneration(ControlNode):
    """FLUX 2 image generation node for Pro and Flex models."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # State to track incoming connections
        self.incoming_connections = {}

        # Model selection
        self.add_parameter(
            Parameter(
                name="model",
                tooltip="FLUX 2 model to use. Pro is faster, Flex allows more control.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="flux-2-pro",
                traits={Options(choices=MODEL_OPTIONS)},
                ui_options={"display_name": "Model"},
            )
        )

        # Prompt input
        self.add_parameter(
            Parameter(
                name="prompt",
                tooltip="Text description of the desired image",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Describe the image you want to generate...",
                },
            )
        )

        # Initialize incoming connection state for prompt parameter
        self.incoming_connections["prompt"] = False

        # Input images for image-to-image generation
        self.add_parameter(
            ParameterList(
                name="input_images",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                default_value=[],
                tooltip="Optional input images for image-to-image generation (supports up to 20MB or 20 megapixels)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"expander": True, "display_name": "Input Images"},
            )
        )

        # Aspect ratio parameter
        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                tooltip="Desired aspect ratio for the generated image",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="1:1",
                traits={Options(choices=ASPECT_RATIO_OPTIONS)},
                ui_options={"display_name": "Aspect Ratio"},
            )
        )

        # Max size parameter (used to calculate width/height from aspect ratio)
        self.add_parameter(
            Parameter(
                name="max_size",
                tooltip="Maximum size in pixels",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=256, max_val=1440)},
                default_value=1024,
                ui_options={"display_name": "Max Size", "step": 32},
            )
        )

        # Image size (calculated from aspect ratio and max size)
        self.add_parameter(
            Parameter(
                name="image_size",
                tooltip="Calculated image size based on max size and aspect ratio",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                default_value="1024x1024",
                ui_options={"display_name": "Image Size"},
            )
        )

        # Seed parameter
        self.add_parameter(
            Parameter(
                name="seed",
                tooltip="Random seed for reproducible results (-1 for random)",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=-1,
                ui_options={"placeholder_text": "Enter seed (optional, -1 for random)"},
            )
        )

        # Prompt upsampling parameter
        self.add_parameter(
            Parameter(
                name="prompt_upsampling",
                tooltip="If enabled, performs upsampling on the prompt for potentially better results",
                type=ParameterTypeBuiltin.BOOL.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=False,
                ui_options={"display_name": "Prompt Upsampling"},
            )
        )

        # Output format parameter
        self.add_parameter(
            Parameter(
                name="output_format",
                tooltip="Desired format of the output image",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="jpeg",
                traits={Options(choices=OUTPUT_FORMAT_OPTIONS)},
                ui_options={"display_name": "Output Format"},
            )
        )

        # Safety tolerance parameter (user-friendly presets)
        self.add_parameter(
            Parameter(
                name="safety_tolerance",
                tooltip="Content moderation level",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="least restrictive",
                traits={Options(choices=SAFETY_TOLERANCE_OPTIONS)},
                ui_options={"display_name": "Safety Tolerance"},
            )
        )

        # Steps parameter (flex only)
        self.add_parameter(
            Parameter(
                name="steps",
                tooltip="Number of inference steps (1-100). Higher = more detail, slower.",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=50,
                traits={Slider(min_val=1, max_val=100)},
                ui_options={"display_name": "Steps", "step": 1, "hide": True},
            )
        )

        # Guidance parameter (flex only)
        self.add_parameter(
            Parameter(
                name="guidance",
                tooltip="Guidance scale (1.5-10). Controls how closely output follows prompt.",
                type=ParameterTypeBuiltin.FLOAT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=4.5,
                traits={Slider(min_val=1.5, max_val=10.0)},
                ui_options={"display_name": "Guidance", "step": 0.1, "hide": True},
            )
        )

        self._output_file = ProjectFileParameter(
            self,
            name="output_file",
            default_filename="flux_image.png",
        )
        self._output_file.add_parameter()

        # Output parameters
        self.add_parameter(
            Parameter(
                name="image",
                tooltip="Generated image with cached data",
                output_type="ImageUrlArtifact",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True},
                settable=False,
            )
        )

        self.add_parameter(
            Parameter(
                name="status",
                tooltip="Generation status and progress",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "pulse_on_run": True},
            )
        )

    def _parse_safety_tolerance(self, value: str | None) -> int:
        """Parse safety tolerance integer from preset string value.

        Args:
            value: One of "least restrictive", "moderate", or "most restrictive"

        Returns:
            Integer value: 5 for least restrictive, 2 for moderate, 0 for most restrictive
        """
        if not value:
            return 5  # Default to least restrictive (API max is 5, not 6)

        if value == "most restrictive":
            return 0
        if value == "moderate":
            return 2
        if value == "least restrictive":
            return 5

        # Fallback to least restrictive
        return 5

    def _calculate_image_size(self, max_size: int, aspect_ratio: str) -> tuple[int, int]:
        """Calculate width and height from max size and aspect ratio.

        Args:
            max_size: Maximum dimension in pixels
            aspect_ratio: Aspect ratio string (e.g., "16:9")

        Returns:
            Tuple of (width, height)
        """
        try:
            width_ratio, height_ratio = map(int, aspect_ratio.split(":"))
        except ValueError:
            raise ValueError("Invalid aspect ratio format. Expected format 'width:height'.")

        # Ensure max_size is an integer
        if not isinstance(max_size, int):
            raise ValueError("max_size must be an integer.")

        # Calculate initial dimensions
        if width_ratio > height_ratio:
            width = max_size
            height = int((max_size / width_ratio) * height_ratio)
        else:
            height = max_size
            width = int((max_size / height_ratio) * width_ratio)

        # Adjust dimensions to be multiples of 32
        width = (width // 32) * 32
        height = (height // 32) * 32

        return width, height

    def _get_api_key(self) -> str:
        """Retrieve the BFL API key from configuration."""
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"BFL API key not found. Please set the {API_KEY_ENV_VAR} environment variable.\n"
                "Get your API key from: https://docs.bfl.ml/"
            )
        return api_key

    def _image_to_base64(self, image_artifact) -> str:
        """Convert ImageArtifact, ImageUrlArtifact, or URL string to base64 string."""
        try:
            # If it's a standard artifact with to_bytes method
            if hasattr(image_artifact, "to_bytes"):
                try:
                    image_bytes = image_artifact.to_bytes()
                    return base64.b64encode(image_bytes).decode("utf-8")
                except Exception:
                    # If to_bytes() fails, continue to URL handling
                    pass

            # Handle ImageUrlArtifact
            if isinstance(image_artifact, ImageUrlArtifact):
                url = image_artifact.value
            # Handle direct URL string
            elif isinstance(image_artifact, str):
                # Validate if string is a proper URL
                if not (image_artifact.startswith("http://") or image_artifact.startswith("https://")):
                    raise ValueError(f"Invalid URL format: {image_artifact}. URL must start with http:// or https://")
                url = image_artifact
            else:
                raise ValueError(
                    f"Unsupported input type: {type(image_artifact).__name__}. Expected ImageArtifact, ImageUrlArtifact, or URL string."
                )

            # Fetch and encode image from URL
            image_bytes = File(url).read_bytes()
            return base64.b64encode(image_bytes).decode("utf-8")

        except Exception as e:
            raise ValueError(f"Failed to convert image to base64: {str(e)}")

    def _process_input_images(self) -> list[str]:
        """Process input images and convert to base64 strings."""
        # Use get_parameter_list_value for ParameterList
        input_images_list = self.get_parameter_list_value("input_images") or []

        if not input_images_list:
            return []

        base64_images = []
        for image_input in input_images_list:
            if image_input and len(base64_images) < MAX_INPUT_IMAGES:
                try:
                    base64_str = self._image_to_base64(image_input)
                    base64_images.append(base64_str)
                except Exception as e:
                    self.append_value_to_parameter("status", f"Warning: Failed to process input image: {str(e)}\n")

        return base64_images

    def _create_request(self, api_key: str, payload: dict[str, Any]) -> str:
        """Create a generation request and return the polling URL."""
        headers = {
            "accept": "application/json",
            "x-key": api_key,
            "Content-Type": "application/json",
        }

        # Get selected model for API endpoint
        model = self.get_parameter_value("model")
        api_url = f"{BFL_API_BASE_URL}/v1/{model}"

        # Debug: Log the request details (without sensitive data)
        debug_payload = payload.copy()
        for key in list(debug_payload.keys()):
            if key.startswith("input_image"):
                if isinstance(debug_payload[key], str):
                    debug_payload[key] = f"<base64_image_{len(payload[key])}_chars>"

        self.append_value_to_parameter(
            "status",
            f"DEBUG - API Request:\nModel: {model}\nPayload keys: {list(payload.keys())}\nPayload (redacted): {debug_payload}\n",
        )

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            # Debug: Log response status and content
            self.append_value_to_parameter("status", f"DEBUG - Request Response Status: {response.status_code}\n")

            response.raise_for_status()
        except ConnectTimeout as e:
            error_msg = (
                f"{self.name}: Connection to BlackForest Labs API timed out after 60 seconds. "
                f"This may indicate network connectivity issues or the API may be temporarily unavailable. "
                f"Please check your internet connection and try again."
            )
            raise ValueError(error_msg) from e
        except Timeout as e:
            error_msg = (
                f"{self.name}: Request to BlackForest Labs API timed out after 60 seconds. "
                f"The API may be experiencing high load. Please try again later."
            )
            raise ValueError(error_msg) from e
        except RequestsConnectionError as e:
            error_msg = (
                f"{self.name}: Failed to connect to BlackForest Labs API at {api_url}. "
                f"This may indicate network connectivity issues. Error: {e!s}"
            )
            raise ValueError(error_msg) from e

        result = response.json()
        self.append_value_to_parameter("status", f"DEBUG - Request Response: {result}\n")

        if "polling_url" not in result:
            raise ValueError(f"{self.name}: Unexpected response format (missing polling_url): {result}")

        return result["polling_url"]

    def _poll_for_result(self, api_key: str, polling_url: str) -> tuple[str, int | None]:
        """Poll for the generation result and return the image URL and seed."""
        headers = {"accept": "application/json", "x-key": api_key}
        self.append_value_to_parameter("status", f"Polling URL: {polling_url}\n")

        max_attempts = 900  # 7.5 minutes with 0.5s intervals
        attempt = 0
        pending_count = 0
        last_status = None

        while attempt < max_attempts:
            time.sleep(0.5)
            attempt += 1

            try:
                response = requests.get(polling_url, headers=headers, timeout=30)
                response.raise_for_status()

                result = response.json()
                status = result.get("status")
                last_status = status

                # Track pending status
                if status == "Pending":
                    pending_count += 1
                else:
                    pending_count = 0

                # Comprehensive debugging - log full response every 10 attempts
                if attempt % 10 == 0 or attempt <= 5:
                    self.append_value_to_parameter(
                        "status",
                        f"DEBUG - Full API response at attempt {attempt}: {result}\n",
                    )

                # Log detailed status for debugging
                self.append_value_to_parameter("status", f"Attempt {attempt}/{max_attempts}: {status}")
                if "result" in result and result.get("result") is not None:
                    self.append_value_to_parameter(
                        "status",
                        f" - Result keys: {list(result.get('result', {}).keys())}",
                    )
                self.append_value_to_parameter("status", "\n")

                # Check if we've been stuck in Pending for too long
                if pending_count > 60:  # 30 seconds of pending
                    self.append_value_to_parameter(
                        "status",
                        f"⚠️ Request has been stuck in 'Pending' status for {pending_count} attempts (30+ seconds).\n",
                    )
                    self.append_value_to_parameter(
                        "status",
                        "This might indicate:\n- API service overload\n- Content safety filters blocking the request\n- Invalid request parameters\n",
                    )
                    self.append_value_to_parameter(
                        "status",
                        "💡 Try: Different prompt, higher safety_tolerance, or check BFL API status\n",
                    )

                    # Try to get more details about why it's stuck
                    if "details" in result and result.get("details"):
                        self.append_value_to_parameter("status", f"API Details: {result.get('details')}\n")

                if status == "Ready":
                    image_url = result.get("result", {}).get("sample")
                    if not image_url:
                        # Try alternative result field names
                        alt_url = result.get("result", {}).get("url") or result.get("result", {}).get("image_url")
                        if alt_url:
                            image_url = alt_url
                        else:
                            self.append_value_to_parameter("status", f"Debug - Full API response: {result}\n")
                            raise ValueError(
                                f"No image URL found in result. Available keys: {list(result.get('result', {}).keys())}"
                            )

                    # Extract the actual seed used by the API
                    api_seed = result.get("result", {}).get("seed")
                    if api_seed:
                        self.append_value_to_parameter("status", f"API used seed: {api_seed}\n")
                    return image_url, api_seed

                elif status in [
                    "Processing",
                    "Queued",
                    "Pending",
                    "Task-queued",
                    "Task-processing",
                ]:
                    # Continue polling for these valid in-progress statuses
                    continue

                elif status == "Request Moderated":
                    moderation_reasons = result.get("details", {}).get("Moderation Reasons", ["Unknown"])
                    self.append_value_to_parameter(
                        "status", f"Request was blocked by content moderation: {', '.join(moderation_reasons)}\n"
                    )
                    self.append_value_to_parameter(
                        "status",
                        "💡 Try: Increase safety_tolerance, use different wording, or avoid restricted content\n",
                    )
                    raise ValueError(
                        f"Request blocked by content moderation: {', '.join(moderation_reasons)}. "
                        f"Try increasing safety_tolerance or modifying your prompt."
                    )

                elif status == "Error" or status == "Failed":
                    error_details = result.get("result", {}).get("error", "Unknown error")
                    self.append_value_to_parameter("status", f"API Error Details: {result}\n")
                    raise ValueError(f"Generation failed with status '{status}': {error_details}")

                else:
                    # Log unknown status for debugging but continue polling
                    self.append_value_to_parameter(
                        "status",
                        f"Unknown status '{status}', continuing to poll. Full response: {result}\n",
                    )
                    continue

            except requests.RequestException as e:
                self.append_value_to_parameter("status", f"Request error on attempt {attempt}: {str(e)}\n")
                if attempt >= max_attempts:
                    raise
                continue

        # Provide more specific timeout message
        if pending_count > 50:
            raise TimeoutError(
                f"Request stuck in 'Pending' status for {pending_count} attempts. This usually indicates API service issues or content safety filters."
            )
        else:
            raise TimeoutError(
                f"Generation timed out after {max_attempts} attempts (7.5 minutes). Last status: {last_status}"
            )

    def _download_image(self, image_url: str) -> bytes:
        """Download image from URL and return bytes."""
        self.append_value_to_parameter("status", f"DEBUG - Downloading from URL: {image_url}\n")
        image_bytes = File(image_url).read_bytes()
        self.append_value_to_parameter("status", f"DEBUG - Downloaded {len(image_bytes)} bytes\n")

        # Log first few bytes to detect if we're getting the same content
        if len(image_bytes) >= 16:
            first_bytes = image_bytes[:16].hex()
            self.append_value_to_parameter("status", f"DEBUG - First 16 bytes: {first_bytes}\n")

        return image_bytes

    def _create_image_artifact(
        self, image_bytes: bytes, output_format: str, api_seed: int | None = None
    ) -> ImageUrlArtifact:
        """Create ImageUrlArtifact by saving to the project file location."""
        try:
            # Generate descriptive name using model, seed, and timestamp
            model = self.get_parameter_value("model").replace("-", "_")
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness

            # Use API seed if available, fallback to user seed, otherwise "random"
            if api_seed is not None:
                seed_str = str(api_seed)
            else:
                user_seed = self.get_parameter_value("seed")
                seed_str = str(user_seed) if user_seed is not None and user_seed != -1 else "random"

            artifact_name = f"bfl_{model}_{seed_str}_{timestamp}"

            self.append_value_to_parameter(
                "status",
                f"DEBUG - Creating artifact: {artifact_name}.{output_format.lower()} ({len(image_bytes)} bytes)\n",
            )

            # Save to project file location and get URL
            dest = self._output_file.build_file()
            saved = dest.write_bytes(image_bytes)

            self.append_value_to_parameter("status", f"DEBUG - File saved to: {saved.location}\n")

            return ImageUrlArtifact(value=saved.location, name=artifact_name)
        except Exception as e:
            raise ValueError(f"Failed to create image artifact: {str(e)}")

    def _poll_and_process_result(self, api_key: str, polling_url: str, output_format: str) -> None:
        """Poll for result, download image, and set output - called via yield."""
        try:
            # Poll for result
            image_url, api_seed = self._poll_for_result(api_key, polling_url)

            # Download image immediately to prevent expiration issues
            self.append_value_to_parameter("status", "Downloading generated image...\n")
            image_bytes = self._download_image(image_url)

            # Create image artifact with proper parameters, using API seed if available
            image_artifact = self._create_image_artifact(image_bytes, output_format, api_seed)

            # Set output
            self.parameter_output_values["image"] = image_artifact

            self.append_value_to_parameter(
                "status",
                f"✅ Generation completed successfully!\nImage URL: {image_url}\n",
            )
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}\n"
            self.append_value_to_parameter("status", error_msg)
            raise

    def after_incoming_connection(
        self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        # Mark the parameter as having an incoming connection
        self.incoming_connections[target_parameter.name] = True

    def after_incoming_connection_removed(
        self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        # Mark the parameter as not having an incoming connection
        self.incoming_connections[target_parameter.name] = False

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # Show/hide flex-specific parameters based on model selection
        if parameter.name == "model":
            if value == "flux-2-flex":
                # Show steps and guidance for flex model
                steps_param = self.get_parameter_by_name("steps")
                if steps_param:
                    steps_param.ui_options["hide"] = False
                    self.publish_update_to_parameter("steps", self.get_parameter_value("steps"))

                guidance_param = self.get_parameter_by_name("guidance")
                if guidance_param:
                    guidance_param.ui_options["hide"] = False
                    self.publish_update_to_parameter("guidance", self.get_parameter_value("guidance"))
            else:
                # Hide steps and guidance for pro model
                steps_param = self.get_parameter_by_name("steps")
                if steps_param:
                    steps_param.ui_options["hide"] = True
                    self.publish_update_to_parameter("steps", self.get_parameter_value("steps"))

                guidance_param = self.get_parameter_by_name("guidance")
                if guidance_param:
                    guidance_param.ui_options["hide"] = True
                    self.publish_update_to_parameter("guidance", self.get_parameter_value("guidance"))

        # Calculate image size when aspect_ratio or max_size changes
        if parameter.name in {"aspect_ratio", "max_size"}:
            aspect_ratio = self.get_parameter_value("aspect_ratio")
            max_size = self.get_parameter_value("max_size")

            width, height = self._calculate_image_size(max_size, aspect_ratio)
            image_size = f"{width}x{height}"

            self.set_parameter_value("image_size", image_size)
            self.publish_update_to_parameter("image_size", image_size)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate node configuration before execution."""
        errors = []

        # Check for prompt
        prompt = self.get_parameter_value("prompt")
        if not (prompt or self.incoming_connections.get("prompt", False)):
            errors.append(
                ValueError(f"{self.name}: Provide a prompt or make a connection to the prompt parameter in this node.")
            )

        # Check for API key
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        if not api_key:
            errors.append(
                ValueError(
                    f"{self.name}: BFL API key not found. Please set the {API_KEY_ENV_VAR} environment variable."
                )
            )

        # Validate seed if provided
        seed = self.get_parameter_value("seed")
        if seed is not None and not isinstance(seed, int):
            errors.append(ValueError(f"{self.name}: Seed must be an integer"))

        return errors if errors else None

    def validate_before_workflow_run(self) -> list[Exception] | None:
        return self.validate_before_node_run()

    def process(self) -> AsyncResult[None]:
        """Non-blocking entry point for Griptape engine."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Generate image using FLUX 2 API."""
        try:
            # Get API key
            api_key = self._get_api_key()

            # Prepare request payload
            output_format = self.get_parameter_value("output_format")
            prompt = self.get_parameter_value("prompt")
            if not prompt:
                raise ValueError("Prompt is required and cannot be empty")

            model = self.get_parameter_value("model")

            # Calculate width and height from aspect ratio
            image_size = self.get_parameter_value("image_size")
            try:
                width, height = map(int, image_size.split("x"))
            except ValueError:
                raise ValueError("Invalid image size format. Expected format 'widthxheight'.")

            # Build base payload with width/height (not aspect_ratio)
            payload = {
                "prompt": prompt.strip(),
                "width": width,
                "height": height,
                "safety_tolerance": self._parse_safety_tolerance(self.get_parameter_value("safety_tolerance")),
                "output_format": output_format,
            }

            # Add flex-specific parameters
            if model == "flux-2-flex":
                payload["steps"] = int(self.get_parameter_value("steps"))
                payload["guidance"] = float(self.get_parameter_value("guidance"))
                # Only flex supports prompt_upsampling
                payload["prompt_upsampling"] = self.get_parameter_value("prompt_upsampling")

            # Add seed if not -1 (random)
            seed = self.get_parameter_value("seed")
            if seed is not None and seed != -1:
                payload["seed"] = int(seed)

            # Process and add input images
            self.append_value_to_parameter("status", "Processing input images...\n")
            base64_images = self._process_input_images()

            if base64_images:
                self.append_value_to_parameter("status", f"Found {len(base64_images)} input image(s)\n")
                # Add images with appropriate keys
                for idx, base64_img in enumerate(base64_images):
                    if idx == 0:
                        payload["input_image"] = base64_img
                    else:
                        payload[f"input_image_{idx + 1}"] = base64_img

            self.append_value_to_parameter("status", "Creating generation request...\n")

            # Create request
            polling_url = self._create_request(api_key, payload)
            self.append_value_to_parameter("status", f"Request created, polling URL: {polling_url}\n")

            # Poll for result using async pattern
            self.append_value_to_parameter("status", "Waiting for generation to complete...\n")
            self._poll_and_process_result(api_key, polling_url, output_format)

        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}\n"
            self.append_value_to_parameter("status", error_msg)
            raise
