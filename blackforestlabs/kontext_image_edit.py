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
from griptape_nodes.exe_types.node_types import ControlNode, BaseNode
from griptape_nodes.traits.options import Options
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

SERVICE = "BlackForest Labs"
API_KEY_ENV_VAR = "BFL_API_KEY"


class KontextImageEdit(ControlNode):
    """FLUX.1 Kontext image editing node."""

    def __init__(self, name: str, metadata: Dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # State to track incoming connections
        self.incoming_connections = {}

        # Input parameters
        self.add_parameter(
            Parameter(
                name="model",
                tooltip="FLUX.1 Kontext model to use. Pro is faster, Max has higher quality.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="flux-kontext-pro",
                traits={Options(choices=["flux-kontext-pro", "flux-kontext-max"])},
                ui_options={"display_name": "Model"},
            )
        )

        self.add_parameter(
            Parameter(
                name="input_image",
                tooltip="Base image to edit",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt",
                tooltip="Text description of edits. For text replacement, use: Replace '[old text]' with '[new text]'",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Describe the edits to make to the image...",
                },
            )
        )

        # Initialize incoming connection state for prompt parameter
        self.incoming_connections["prompt"] = False


        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                tooltip="Desired aspect ratio for the edited image. Supports ratios from 3:7 to 7:3.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="1:1",
                traits={
                    Options(
                        choices=[
                            "7:3",
                            "16:9",
                            "3:2",
                            "4:3",
                            "1:1",
                            "3:4",
                            "2:3",
                            "9:16",
                            "3:7",
                        ]
                    )
                },
                ui_options={"display_name": "Aspect Ratio"},
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
                name="safety_tolerance",
                tooltip="Moderation level. 0 = most strict, 2 = balanced",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=2,
                traits={Options(choices=[0, 1, 2])},
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
                name="edited_image",
                tooltip="Edited image with cached data",
                output_type="ImageUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="status",
                tooltip="Editing status and progress",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "pulse_on_run": True},
            )
        )

    def _get_api_key(self) -> str:
        """Retrieve the BFL API key from configuration."""
        api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"BFL API key not found. Please set the {API_KEY_ENV_VAR} environment variable.\n"
                "Get your API key from: https://docs.bfl.ml/"
            )
        return api_key

    def _image_to_base64(self, image_artifact) -> str:
        """Convert ImageArtifact or ImageUrlArtifact to base64 string."""
        try:
            # Try the standard to_bytes() method first
            try:
                image_bytes = image_artifact.to_bytes()
                return base64.b64encode(image_bytes).decode("utf-8")
            except (AttributeError, Exception):
                # If to_bytes() fails, try to fetch from URL for ImageUrlArtifact
                if isinstance(image_artifact, ImageUrlArtifact):
                    response = requests.get(image_artifact.value, timeout=30)
                    response.raise_for_status()
                    image_bytes = response.content
                    return base64.b64encode(image_bytes).decode("utf-8")
                else:
                    # Re-raise the original exception if it's not an ImageUrlArtifact
                    raise
        except Exception as e:
            raise ValueError(f"Failed to convert image to base64: {str(e)}")

    def _create_request(self, api_key: str, payload: Dict[str, Any]) -> tuple[str, str]:
        """Create an editing request and return the request ID and polling URL."""
        headers = {
            "accept": "application/json",
            "x-key": api_key,
            "Content-Type": "application/json",
        }

        # Get selected model for API endpoint
        model = self.get_parameter_value("model")

        # Debug: Log the request details (without sensitive data)
        debug_payload = payload.copy()
        if "input_image" in debug_payload:
            debug_payload["input_image"] = (
                f"<base64_image_{len(payload['input_image'])}_chars>"
            )

        self.append_value_to_parameter(
            "status",
            f"DEBUG - API Request:\nModel: {model}\nPayload keys: {list(payload.keys())}\nPayload (redacted): {debug_payload}\n",
        )

        response = requests.post(
            f"https://api.us1.bfl.ai/v1/{model}",
            headers=headers,
            json=payload,
            timeout=30,
        )

        # Debug: Log response status and content
        self.append_value_to_parameter(
            "status", f"DEBUG - Request Response Status: {response.status_code}\n"
        )

        response.raise_for_status()

        result = response.json()
        self.append_value_to_parameter(
            "status", f"DEBUG - Request Response: {result}\n"
        )

        if "id" not in result:
            raise ValueError(f"Unexpected response format: {result}")

        request_id = result["id"]
        polling_url = result.get("polling_url")  # Get polling URL if provided

        return request_id, polling_url

    def _poll_for_result(
        self, api_key: str, request_id: str, polling_url: Optional[str] = None
    ) -> str:
        """Poll for the editing result and return the image URL."""
        headers = {"accept": "application/json", "x-key": api_key}

        # Use provided polling URL or construct default one
        url = (
            polling_url
            if polling_url
            else f"https://api.us1.bfl.ai/v1/get_result?id={request_id}"
        )
        self.append_value_to_parameter("status", f"Polling URL: {url}\n")

        max_attempts = 300  # Increased to 7.5 minutes with 1.5s intervals
        attempt = 0
        pending_count = 0  # Track how long we've been stuck in Pending
        last_status = None

        while attempt < max_attempts:
            time.sleep(1.5)
            attempt += 1

            try:
                response = requests.get(url, headers=headers, timeout=30)
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
                self.append_value_to_parameter(
                    "status", f"Attempt {attempt}/{max_attempts}: {status}"
                )
                if "result" in result and result.get("result") is not None:
                    self.append_value_to_parameter(
                        "status",
                        f" - Result keys: {list(result.get('result', {}).keys())}",
                    )
                self.append_value_to_parameter("status", "\n")

                # Check if we've been stuck in Pending for too long
                if pending_count > 60:  # 1.5 minutes of pending
                    self.append_value_to_parameter(
                        "status",
                        f"⚠️ Request has been stuck in 'Pending' status for {pending_count} attempts (1.5+ minutes).\n",
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
                        self.append_value_to_parameter(
                            "status", f"API Details: {result.get('details')}\n"
                        )

                if status == "Ready":
                    image_url = result.get("result", {}).get("sample")
                    if not image_url:
                        # Try alternative result field names
                        alt_url = result.get("result", {}).get("url") or result.get(
                            "result", {}
                        ).get("image_url")
                        if alt_url:
                            image_url = alt_url
                        else:
                            self.append_value_to_parameter(
                                "status", f"Debug - Full API response: {result}\n"
                            )
                            raise ValueError(
                                f"No image URL found in result. Available keys: {list(result.get('result', {}).keys())}"
                            )
                    return image_url

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
                        "status", "💡 Try: Increase safety_tolerance, use different wording, or avoid restricted content (police, violence, etc.)\n"
                    )
                    raise ValueError(
                        f"Request blocked by content moderation: {', '.join(moderation_reasons)}. "
                        f"Try increasing safety_tolerance or modifying your prompt to avoid restricted content."
                    )

                elif status == "Error" or status == "Failed":
                    error_details = result.get("result", {}).get(
                        "error", "Unknown error"
                    )
                    self.append_value_to_parameter(
                        "status", f"API Error Details: {result}\n"
                    )
                    raise ValueError(
                        f"Editing failed with status '{status}': {error_details}"
                    )

                else:
                    # Log unknown status for debugging but continue polling
                    self.append_value_to_parameter(
                        "status",
                        f"Unknown status '{status}', continuing to poll. Full response: {result}\n",
                    )
                    continue

            except requests.RequestException as e:
                self.append_value_to_parameter(
                    "status", f"Request error on attempt {attempt}: {str(e)}\n"
                )
                if attempt >= max_attempts:
                    raise
                continue

        # Provide more specific timeout message
        if pending_count > 50:
            raise TimeoutError(
                f"Request stuck in 'Pending' status for {pending_count} attempts. This usually indicates API service issues or content safety filters. Try adjusting safety_tolerance or simplifying the prompt."
            )
        else:
            raise TimeoutError(
                f"Editing timed out after {max_attempts} attempts (7.5 minutes). Last status: {last_status}"
            )

    def _download_image(self, image_url: str) -> bytes:
        """Download image from URL and return bytes."""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise ValueError(f"Failed to download image from URL: {str(e)}")

    def _create_image_artifact(
        self, image_bytes: bytes, output_format: str
    ) -> ImageUrlArtifact:
        """Create ImageUrlArtifact using StaticFilesManager for efficient storage."""
        try:
            # Generate unique filename with timestamp and hash
            import hashlib

            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            content_hash = hashlib.md5(image_bytes).hexdigest()[
                :8
            ]  # Short hash of content
            filename = (
                f"kontext_image_edit_{timestamp}_{content_hash}.{output_format.lower()}"
            )

            # Save to managed file location and get URL
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                image_bytes, filename
            )

            return ImageUrlArtifact(
                value=static_url, name=f"kontext_image_edit_{timestamp}"
            )
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

        # Check for API key
        api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)
        if not api_key:
            errors.append(
                ValueError(
                    f"{self.name}: BFL API key not found. Please set the {API_KEY_ENV_VAR} environment variable."
                )
            )

        # Check for input image
        input_image = self.get_parameter_value("input_image")
        if not input_image:
            errors.append(ValueError(f"{self.name}: Input image is required"))

        # Check for prompt
        prompt = self.get_parameter_value("prompt")
        if not (prompt or self.incoming_connections.get("prompt", False)):
            errors.append(
                ValueError(f"{self.name}: Provide a prompt or make a connection to the prompt parameter in this node.")
            )

        # Validate seed if provided
        seed = self.get_parameter_value("seed")
        if seed is not None and not isinstance(seed, int):
            errors.append(ValueError(f"{self.name}: Seed must be an integer"))

        return errors if errors else None

    def validate_before_workflow_run(self) -> list[Exception] | None:
        return self.validate_before_node_run()

    def process(self) -> None:
        """Edit image using FLUX.1 Kontext API."""
        try:
            # Get API key
            api_key = self._get_api_key()

            # Get input image and convert to base64
            input_image = self.get_parameter_value("input_image")
            self.append_value_to_parameter(
                "status", "Converting input image to base64...\n"
            )
            input_image_base64 = self._image_to_base64(input_image)

            # Prepare request payload
            output_format = self.get_parameter_value("output_format")
            prompt = self.get_parameter_value("prompt")
            if not prompt:
                raise ValueError("Prompt is required and cannot be empty")

            payload = {
                "prompt": prompt.strip(),
                "input_image": input_image_base64,
                "aspect_ratio": self.get_parameter_value("aspect_ratio"),
                "prompt_upsampling": self.get_parameter_value("prompt_upsampling"),
                "safety_tolerance": int(self.get_parameter_value("safety_tolerance")),
                "output_format": output_format,
            }

            # Add seed if provided
            seed = self.get_parameter_value("seed")
            if seed is not None and seed != 0:  # Don't send seed=0
                payload["seed"] = int(seed)

            self.append_value_to_parameter("status", "Creating editing request...\n")

            # Create request
            request_id, polling_url = self._create_request(api_key, payload)
            self.append_value_to_parameter(
                "status", f"Request created with ID: {request_id}\n"
            )

            # Poll for result
            self.append_value_to_parameter(
                "status", "Waiting for editing to complete...\n"
            )
            image_url = self._poll_for_result(api_key, request_id, polling_url)

            # Download image immediately to prevent expiration issues
            self.append_value_to_parameter("status", "Downloading edited image...\n")
            image_bytes = self._download_image(image_url)

            # Create image artifact with proper parameters
            image_artifact = self._create_image_artifact(image_bytes, output_format)

            # Set output
            self.parameter_output_values["edited_image"] = image_artifact

            self.append_value_to_parameter(
                "status",
                f"✅ Editing completed successfully!\nImage URL: {image_url}\n",
            )

        except Exception as e:
            error_msg = f"❌ Editing failed: {str(e)}\n"
            self.append_value_to_parameter("status", error_msg)
            raise
