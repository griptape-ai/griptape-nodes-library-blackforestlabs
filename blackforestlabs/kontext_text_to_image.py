import base64
import io
import time
from typing import Any, Dict, Optional

import requests
from PIL import Image
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.traits.options import Options
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

SERVICE = "BlackForest Labs"
API_KEY_ENV_VAR = "BFL_API_KEY"


class KontextTextToImage(ControlNode):
    """FLUX.1 Kontext text-to-image generation node."""

    def __init__(self, name: str, metadata: Dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Input parameters
        self.add_parameter(
            Parameter(
                name="model",
                tooltip="FLUX.1 Kontext model to use. Pro is faster, Max has higher quality.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="flux-kontext-pro",
                traits={Options(choices=["flux-kontext-pro", "flux-kontext-max"])},
                ui_options={"display_name": "Model"}
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt",
                tooltip="Text description of the desired image",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": True, "placeholder_text": "Describe the image you want to generate..."}
            )
        )

        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                tooltip="Desired aspect ratio for the generated image. Supports ratios from 3:7 to 7:3.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="1:1",
                traits={Options(choices=[
                    "3:7", "9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9", "7:3"
                ])},
                ui_options={"display_name": "Aspect Ratio"}
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                tooltip="Seed for reproducibility. Leave empty for random generation.",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "Enter seed (optional)"}
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt_upsampling",
                tooltip="If enabled, performs upsampling on the prompt for potentially better results",
                type=ParameterTypeBuiltin.BOOL.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=False,
                ui_options={"display_name": "Prompt Upsampling"}
            )
        )

        self.add_parameter(
            Parameter(
                name="safety_tolerance",
                tooltip="Moderation level. 0 = most strict, 2 = least strict",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=2,
                traits={Options(choices=["0", "1", "2"])},
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
                name="image",
                tooltip="Generated image with cached data",
                output_type="ImageUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT}
            )
        )

        self.add_parameter(
            Parameter(
                name="status",
                tooltip="Generation status and progress",
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

    def _create_request(self, api_key: str, payload: Dict[str, Any]) -> str:
        """Create a generation request and return the request ID."""
        headers = {
            "accept": "application/json",
            "x-key": api_key,
            "Content-Type": "application/json"
        }

        # Get selected model for API endpoint
        model = self.get_parameter_value("model")
        
        response = requests.post(
            f"https://api.us1.bfl.ai/v1/{model}",
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
        """Poll for the generation result and return the image URL."""
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
                    raise ValueError(f"Generation failed with status '{status}': {result}")

            except requests.RequestException as e:
                self.append_value_to_parameter("status", f"Request error on attempt {attempt}: {str(e)}\n")
                if attempt >= max_attempts:
                    raise

        raise TimeoutError(f"Generation timed out after {max_attempts} attempts")

    def _download_image(self, image_url: str) -> bytes:
        """Download image from URL and return bytes."""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise ValueError(f"Failed to download image from URL: {str(e)}")

    def _create_image_artifact(self, image_bytes: bytes, output_format: str) -> ImageUrlArtifact:
        """Create ImageUrlArtifact using StaticFilesManager for efficient storage."""
        try:
            # Generate unique filename with timestamp and hash
            import hashlib
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            content_hash = hashlib.md5(image_bytes).hexdigest()[:8]  # Short hash of content
            filename = f"kontext_text_to_image_{timestamp}_{content_hash}.{output_format.lower()}"
            
            # Save to managed file location and get URL
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)
            
            return ImageUrlArtifact(
                value=static_url,
                name=f"kontext_text_to_image_{timestamp}"
            )
        except Exception as e:
            raise ValueError(f"Failed to create image artifact: {str(e)}")

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate node configuration before execution."""
        errors = []

        # Check for API key
        api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)
        if not api_key:
            errors.append(ValueError(f"BFL API key not found. Please set the {API_KEY_ENV_VAR} environment variable."))

        # Check for prompt
        prompt = self.get_parameter_value("prompt")
        if not prompt or not prompt.strip():
            errors.append(ValueError("Prompt is required and cannot be empty"))

        # Validate seed if provided
        seed = self.get_parameter_value("seed")
        if seed is not None and not isinstance(seed, int):
            errors.append(ValueError("Seed must be an integer"))

        return errors if errors else None

    def process(self) -> None:
        """Generate image using FLUX.1 Kontext API."""
        try:
            # Get API key
            api_key = self._get_api_key()

            # Prepare request payload
            output_format = self.get_parameter_value("output_format")
            payload = {
                "prompt": self.get_parameter_value("prompt").strip(),
                "aspect_ratio": self.get_parameter_value("aspect_ratio"),
                "prompt_upsampling": self.get_parameter_value("prompt_upsampling"),
                "safety_tolerance": int(self.get_parameter_value("safety_tolerance")),
                "output_format": output_format
            }

            # Add seed if provided
            seed = self.get_parameter_value("seed")
            if seed is not None:
                payload["seed"] = int(seed)

            self.append_value_to_parameter("status", "Creating generation request...\n")

            # Create request
            request_id = self._create_request(api_key, payload)
            self.append_value_to_parameter("status", f"Request created with ID: {request_id}\n")

            # Poll for result
            self.append_value_to_parameter("status", "Waiting for generation to complete...\n")
            image_url = self._poll_for_result(api_key, request_id)

            # Download image immediately to prevent expiration issues
            self.append_value_to_parameter("status", "Downloading generated image...\n")
            image_bytes = self._download_image(image_url)

            # Create image artifact with proper parameters
            image_artifact = self._create_image_artifact(image_bytes, output_format)
            
            # Set output
            self.parameter_output_values["image"] = image_artifact

            self.append_value_to_parameter("status", f"✅ Generation completed successfully!\nImage URL: {image_url}\n")

        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}\n"
            self.append_value_to_parameter("status", error_msg)
            raise 