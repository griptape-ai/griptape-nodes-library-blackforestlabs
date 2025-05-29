import base64
import io
import time
from typing import Any, Dict

import requests
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.traits.options import Options

SERVICE = "BlackForest Labs"
API_KEY_ENV_VAR = "BFL_API_KEY"


class KontextImageEdit(ControlNode):
    """FLUX.1 Kontext image editing node for modifying existing images."""

    def __init__(self, name: str, metadata: Dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Input parameters
        self.add_parameter(
            Parameter(
                name="input_image",
                tooltip="Input image to edit",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Input Image"}
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt",
                tooltip="Text description of what you want to edit on the image. Use quotes for text replacement: Replace '[old text]' with '[new text]'",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True, 
                    "placeholder_text": "Describe what you want to edit on the image..."
                }
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                tooltip="Seed for reproducibility. Leave empty for random editing.",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "Enter seed (optional)"}
            )
        )

        self.add_parameter(
            Parameter(
                name="safety_tolerance",
                tooltip="Moderation level. 0 = most strict, 6 = most permissive",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=2,
                traits={Options(choices=["0", "1", "2", "3", "4", "5", "6"])},
                ui_options={"display_name": "Safety Tolerance"}
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
                ui_options={"display_name": "Output Format"}
            )
        )

        # Output parameters
        self.add_parameter(
            Parameter(
                name="edited_image",
                tooltip="Edited image",
                type="ImageUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT}
            )
        )

        self.add_parameter(
            Parameter(
                name="status",
                tooltip="Editing status and progress",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "pulse_on_run": True}
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

    def _image_to_base64(self, image_artifact: ImageArtifact | ImageUrlArtifact) -> str:
        """Convert image artifact to base64 string."""
        try:
            # Get the image bytes from the artifact
            image_bytes = image_artifact.to_bytes()
            
            # Convert to base64
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            return base64_string
            
        except Exception as e:
            raise ValueError(f"Failed to convert image to base64: {str(e)}")

    def _create_request(self, api_key: str, payload: Dict[str, Any]) -> str:
        """Create an image editing request and return the request ID."""
        headers = {
            "accept": "application/json",
            "x-key": api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.us1.bfl.ai/v1/flux-kontext-pro",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        if "id" not in result:
            raise ValueError(f"Unexpected response format: {result}")

        return result["id"]

    def _poll_for_result(self, api_key: str, request_id: str) -> str:
        """Poll for the editing result and return the image URL."""
        headers = {
            "accept": "application/json",
            "x-key": api_key
        }

        max_attempts = 120  # 3 minutes with 1.5s intervals
        attempt = 0

        while attempt < max_attempts:
            time.sleep(1.5)
            attempt += 1

            try:
                response = requests.get(
                    f"https://api.us1.bfl.ai/v1/get_result?id={request_id}",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()

                result = response.json()
                status = result.get("status")

                self.append_value_to_parameter("status", f"Attempt {attempt}: {status}\n")

                if status == "Ready":
                    image_url = result.get("result", {}).get("sample")
                    if not image_url:
                        raise ValueError(f"No image URL in result: {result}")
                    return image_url

                elif status in ["Processing", "Queued", "Pending"]:
                    continue

                else:
                    raise ValueError(f"Editing failed with status '{status}': {result}")

            except requests.RequestException as e:
                self.append_value_to_parameter("status", f"Request error on attempt {attempt}: {str(e)}\n")
                if attempt >= max_attempts:
                    raise

        raise TimeoutError(f"Image editing timed out after {max_attempts} attempts")

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate node configuration before execution."""
        errors = []

        # Check for API key
        api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)
        if not api_key:
            errors.append(ValueError(f"BFL API key not found. Please set the {API_KEY_ENV_VAR} environment variable."))

        # Check for input image
        input_image = self.get_parameter_value("input_image")
        if not input_image:
            errors.append(ValueError("Input image is required"))

        # Check for prompt
        prompt = self.get_parameter_value("prompt")
        if not prompt or not prompt.strip():
            errors.append(ValueError("Edit prompt is required and cannot be empty"))

        # Validate seed if provided
        seed = self.get_parameter_value("seed")
        if seed is not None and not isinstance(seed, int):
            errors.append(ValueError("Seed must be an integer"))

        return errors if errors else None

    def process(self) -> None:
        """Edit image using FLUX.1 Kontext API."""
        try:
            # Get API key
            api_key = self._get_api_key()

            # Get and convert input image
            input_image = self.get_parameter_value("input_image")
            self.append_value_to_parameter("status", "Converting input image to base64...\n")
            base64_image = self._image_to_base64(input_image)

            # Prepare request payload
            payload = {
                "prompt": self.get_parameter_value("prompt").strip(),
                "input_image": base64_image,
                "safety_tolerance": int(self.get_parameter_value("safety_tolerance")),
                "output_format": self.get_parameter_value("output_format")
            }

            # Add seed if provided
            seed = self.get_parameter_value("seed")
            if seed is not None:
                payload["seed"] = int(seed)

            self.append_value_to_parameter("status", "Creating image editing request...\n")

            # Create request
            request_id = self._create_request(api_key, payload)
            self.append_value_to_parameter("status", f"Request created with ID: {request_id}\n")

            # Poll for result
            self.append_value_to_parameter("status", "Waiting for image editing to complete...\n")
            edited_image_url = self._poll_for_result(api_key, request_id)

            # Create image artifact
            edited_image_artifact = ImageUrlArtifact(value=edited_image_url, name="edited_image")
            self.parameter_output_values["edited_image"] = edited_image_artifact

            self.append_value_to_parameter("status", f"✅ Image editing completed successfully!\nEdited image URL: {edited_image_url}\n")

        except Exception as e:
            error_msg = f"❌ Image editing failed: {str(e)}\n"
            self.append_value_to_parameter("status", error_msg)
            raise 