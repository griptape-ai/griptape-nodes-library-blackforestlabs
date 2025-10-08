import base64
import io
import time
from typing import Any, Dict, Optional

import requests
from PIL import Image
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import ControlNode, BaseNode, AsyncResult
from griptape_nodes.traits.options import Options
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.slider import Slider

SERVICE = "BlackForest Labs"
API_KEY_ENV_VAR = "BFL_API_KEY"


class TextToImage(ControlNode):
    """FLUX text-to-image generation node for Pro, Dev, Ultra, and Kontext models."""

    def __init__(self, name: str, metadata: Dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # State to track incoming connections
        self.incoming_connections = {}

        # Input parameters
        self.add_parameter(
            Parameter(
                name="model",
                tooltip="FLUX model to use. Ultra has highest quality, Pro is balanced, Dev is open-source.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="flux-pro-1.1",
                traits={
                    Options(
                        choices=[
                            "flux-kontext-pro",
                            "flux-kontext-max",
                            "flux-pro-1.1-ultra",
                            "flux-pro-1.1",
                            "flux-pro",
                            "flux-dev",
                        ]
                    )
                },
                ui_options={"display_name": "Model"},
            )
        )

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

        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                tooltip="Desired aspect ratio for the generated image.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="1:1",
                traits={
                    Options(
                        choices=[
                            "3:7",
                            "9:21",
                            "9:16",
                            "2:3",
                            "3:4",
                            "1:1",
                            "4:3",
                            "3:2",
                            "16:9",
                            "7:3",
                            "21:9",
                        ]
                    )
                },
                ui_options={"display_name": "Aspect Ratio"},
            )
        )

        self.add_parameter(
            Parameter(
                name="max_size",
                tooltip="Maximum size in pixels.",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=256, max_val=1440)},
                default_value=1024,
                ui_options={"display_name": "Max Size", "step": 32},
            )
        )

        self.add_parameter(
            Parameter(
                name="image_size",
                tooltip="Calculated image size based on max size and aspect ratio.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                default_value="1024x1024",
                ui_options={"display_name": "Image Size"},
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                tooltip="Seed for reproducibility. Leave empty for random generation.",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "Enter seed (optional)"},
            )
        )

        # Kontext-only (ignored for classic FLUX models)
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

        self.add_parameter(
            Parameter(
                name="raw",
                tooltip="Generate less processed, more natural-looking images",
                type=ParameterTypeBuiltin.BOOL.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=False,
                ui_options={"display_name": "Raw Mode"},
            )
        )

        self.add_parameter(
            Parameter(
                name="safety_tolerance",
                tooltip="Moderation level. 0 = most strict, 6 = least strict",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=2,
                traits={Options(choices=[0, 1, 2, 3, 4, 5, 6])},
                ui_options={"display_name": "Safety Tolerance"},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_format",
                tooltip="Desired format of the output image",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="jpeg",
                traits={Options(choices=["jpeg", "png"])},
                ui_options={"display_name": "Output Format"},
            )
        )

        # Output parameters
        self.add_parameter(
            Parameter(
                name="image",
                tooltip="Generated image with cached data",
                output_type="ImageUrlArtifact",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True},
                settable=False,  # Ensures this serializes on save, but don't let user set it.
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

    def _calculate_image_size(self, max_size: int, aspect_ratio: str) -> str:
        # Parse the aspect ratio
        try:
            width_ratio, height_ratio = map(int, aspect_ratio.split(":"))
        except ValueError:
            raise ValueError(
                "Invalid aspect ratio format. Expected format 'width:height'."
            )

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

        return f"{width}x{height}"

    def _get_api_key(self) -> str:
        """Retrieve the BFL API key from configuration."""
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"BFL API key not found. Please set the {API_KEY_ENV_VAR} environment variable.\n"
                "Get your API key from: https://docs.bfl.ml/"
            )
        return api_key

    def _create_request(self, api_key: str, payload: Dict[str, Any]) -> str:
        """Create a generation request and return the request ID."""
        headers = {
            "accept": "application/json",
            "x-key": api_key,
            "Content-Type": "application/json",
        }

        # Get selected model for API endpoint
        model = self.get_parameter_value("model")

        response = requests.post(
            f"https://api.us1.bfl.ai/v1/{model}",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        if "id" not in result:
            raise ValueError(f"Unexpected response format: {result}")

        return result["id"]

    def _poll_and_process_result(self, api_key: str, request_id: str, output_format: str) -> None:
        """Poll for the generation result with backoff, download image, and set output - called via yield."""
        try:
            headers = {"accept": "application/json", "x-key": api_key}
            max_attempts = 120
            attempt = 0
            consecutive_500_errors = 0
            base_sleep = 1.5
            image_url = None
            api_seed = None

            while attempt < max_attempts and image_url is None:
                # Exponential backoff with jitter similar to Kontext node
                import random
                sleep_time = min(base_sleep * (2 ** min(attempt // 10, 4)), 10)
                sleep_time += random.uniform(0, 0.5)
                time.sleep(sleep_time)

                attempt += 1

                try:
                    response = requests.get(
                        f"https://api.us1.bfl.ai/v1/get_result?id={request_id}",
                        headers=headers,
                        timeout=30,
                    )

                    if response.status_code == 500:
                        consecutive_500_errors += 1
                        self.append_value_to_parameter(
                            "status", f"Attempt {attempt}: Server error (500) - #{consecutive_500_errors} consecutive\n"
                        )
                        if consecutive_500_errors >= 10:
                            raise ValueError(
                                "API appears to have persistent server issues (10+ consecutive 500 errors). "
                                f"This may be a problem with the {self.get_parameter_value('model')} model endpoint. "
                                "Try a different model or retry later."
                            )
                        continue
                    elif response.status_code == 429:
                        self.append_value_to_parameter(
                            "status", f"Attempt {attempt}: Rate limited, waiting longer...\n"
                        )
                        time.sleep(5)
                        continue
                    elif response.status_code != 200:
                        response.raise_for_status()

                    consecutive_500_errors = 0

                    result = response.json()
                    status = result.get("status")
                    self.append_value_to_parameter("status", f"Attempt {attempt}: {status}\n")

                    if status == "Ready":
                        image_url = result.get("result", {}).get("sample")
                        if not image_url:
                            raise ValueError(f"No image URL in result: {result}")

                        api_seed = result.get("result", {}).get("seed")
                        if api_seed:
                            self.append_value_to_parameter("status", f"API used seed: {api_seed}\n")
                        break
                    elif status in ["Processing", "Queued", "Pending"]:
                        continue
                    else:
                        raise ValueError(
                            f"Generation failed with status '{status}': {result}"
                        )

                except requests.RequestException as e:
                    if not (hasattr(e, 'response') and e.response and e.response.status_code == 500):
                        self.append_value_to_parameter(
                            "status", f"Request error on attempt {attempt}: {str(e)}\n"
                        )
                    if attempt >= max_attempts:
                        raise

            if image_url is None:
                raise TimeoutError(f"Generation timed out after {max_attempts} attempts")

            self.append_value_to_parameter("status", "Downloading generated image...\n")
            image_bytes = self._download_image(image_url)
            image_artifact = self._create_image_artifact(image_bytes, output_format, api_seed)
            self.parameter_output_values["image"] = image_artifact
            self.append_value_to_parameter(
                "status", f"✅ Generation completed successfully!\nImage URL: {image_url}\n"
            )
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}\n"
            self.append_value_to_parameter("status", error_msg)
            raise

    def _download_image(self, image_url: str) -> bytes:
        """Download image from URL and return bytes."""
        try:
            self.append_value_to_parameter("status", f"DEBUG - Downloading from URL: {image_url}\n")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            self.append_value_to_parameter("status", f"DEBUG - Downloaded {len(image_bytes)} bytes\n")
            
            # Log first few bytes to detect if we're getting the same content
            if len(image_bytes) >= 16:
                first_bytes = image_bytes[:16].hex()
                self.append_value_to_parameter("status", f"DEBUG - First 16 bytes: {first_bytes}\n")
            
            return image_bytes
        except Exception as e:
            raise ValueError(f"Failed to download image from URL: {str(e)}")

    def _create_image_artifact(
        self, image_bytes: bytes, output_format: str, api_seed: int = None
    ) -> ImageUrlArtifact:
        """Create ImageUrlArtifact using StaticFilesManager for efficient storage."""
        try:
            # Generate descriptive filename using model, seed, and timestamp
            model = self.get_parameter_value("model").replace("-", "_")  # Replace hyphens for filename
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            
            # Use API seed if available, fallback to user seed, otherwise "random" 
            if api_seed is not None:
                seed_str = str(api_seed)
            else:
                user_seed = self.get_parameter_value("seed")
                seed_str = str(user_seed) if user_seed is not None else "random"
            
            filename = f"bfl_{model}_{seed_str}_{timestamp}.{output_format.lower()}"
            
            self.append_value_to_parameter("status", f"DEBUG - Creating artifact: {filename} ({len(image_bytes)} bytes)\n")

            # Save to managed file location and get URL
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                image_bytes, filename
            )
            
            self.append_value_to_parameter("status", f"DEBUG - Static URL created: {static_url}\n")

            return ImageUrlArtifact(value=static_url, name=f"bfl_{model}_{seed_str}_{timestamp}")
        except Exception as e:
            raise ValueError(f"Failed to create image artifact: {str(e)}")

    def after_incoming_connection(self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter) -> None:
        # Mark the parameter as having an incoming connection
        self.incoming_connections[target_parameter.name] = True

    def after_incoming_connection_removed(self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter) -> None:
        # Mark the parameter as not having an incoming connection
        self.incoming_connections[target_parameter.name] = False

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
        """Generate image using FLUX API."""

        try:
            # Get API key
            api_key = self._get_api_key()

            # Prepare request payload
            output_format = self.get_parameter_value("output_format")

            prompt = self.get_parameter_value("prompt")
            if not prompt:
                raise ValueError("Prompt is required and cannot be empty")

            model = self.get_parameter_value("model")

            # Build payload conditionally depending on model family
            if isinstance(model, str) and model.startswith("flux-kontext"):
                # Kontext endpoints expect aspect_ratio (and ignore width/height and raw)
                payload = {
                    "prompt": prompt.strip(),
                    "aspect_ratio": self.get_parameter_value("aspect_ratio"),
                    "prompt_upsampling": self.get_parameter_value("prompt_upsampling"),
                    "safety_tolerance": int(self.get_parameter_value("safety_tolerance")),
                    "output_format": output_format,
                }
            else:
                # Classic FLUX endpoints expect explicit width/height and support raw
                image_size = self.get_parameter_value("image_size")
                try:
                    width, height = map(int, image_size.split("x"))
                except ValueError:
                    raise ValueError(
                        "Invalid image size format. Expected format 'widthxheight'."
                    )

                payload = {
                    "prompt": prompt.strip(),
                    "width": width,
                    "height": height,
                    "raw": self.get_parameter_value("raw"),
                    "safety_tolerance": int(self.get_parameter_value("safety_tolerance")),
                    "output_format": output_format,
                }

            # Add seed if provided
            seed = self.get_parameter_value("seed")
            if seed is not None:
                payload["seed"] = int(seed)

            self.append_value_to_parameter("status", "Creating generation request...\n")

            # Create request
            request_id = self._create_request(api_key, payload)
            self.append_value_to_parameter(
                "status", f"Request created with ID: {request_id}\n"
            )

            # Poll for result using async pattern
            self.append_value_to_parameter(
                "status", "Waiting for generation to complete...\n"
            )
            self._poll_and_process_result(api_key, request_id, output_format)

        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}\n"
            self.append_value_to_parameter("status", error_msg)
            raise

    def after_value_set(
        self, parameter: Parameter, value: Any
    ) -> None:
        # Check if the updated parameter is aspect_ratio or max_size
        if parameter.name in {"aspect_ratio", "max_size"}:
            # Get current values of aspect_ratio and max_size
            aspect_ratio = self.get_parameter_value("aspect_ratio")
            max_size = self.get_parameter_value("max_size")

            # Calculate the image size
            image_size = self._calculate_image_size(max_size, aspect_ratio)

            # Set the image_size parameter
            self.set_parameter_value("image_size", image_size)
            self.publish_update_to_parameter("image_size", image_size)
