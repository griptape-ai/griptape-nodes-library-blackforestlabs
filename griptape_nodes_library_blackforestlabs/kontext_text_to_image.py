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

SERVICE = "BlackForest Labs"
API_KEY_ENV_VAR = "BFL_API_KEY"


class KontextTextToImage(ControlNode):
    """FLUX.1 Kontext text-to-image generation node."""

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
                tooltip="Desired aspect ratio for the generated image. Supports ratios from 3:7 to 7:3.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="1:1",
                traits={
                    Options(
                        choices=[
                            "3:7",
                            "9:16",
                            "2:3",
                            "3:4",
                            "1:1",
                            "4:3",
                            "3:2",
                            "16:9",
                            "7:3",
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
                tooltip="Moderation level. 0 = most strict, 6 = most permissive",
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

    def _poll_for_result(self, api_key: str, request_id: str) -> str:
        """Poll for the generation result and return the image URL."""
        headers = {"accept": "application/json", "x-key": api_key}

        max_attempts = 120  # 3 minutes with exponential backoff
        attempt = 0
        consecutive_500_errors = 0
        base_sleep = 1.5

        while attempt < max_attempts:
            # Exponential backoff with jitter
            import random
            sleep_time = min(base_sleep * (2 ** min(attempt // 10, 4)), 10)  # Cap at 10s
            sleep_time += random.uniform(0, 0.5)  # Add jitter
            time.sleep(sleep_time)
            
            attempt += 1

            try:
                response = requests.get(
                    f"https://api.us1.bfl.ai/v1/get_result?id={request_id}",
                    headers=headers,
                    timeout=30,
                )
                
                # Handle different status codes differently
                if response.status_code == 500:
                    consecutive_500_errors += 1
                    self.append_value_to_parameter(
                        "status", f"Attempt {attempt}: Server error (500) - #{consecutive_500_errors} consecutive\n"
                    )
                    
                    # If we get too many consecutive 500 errors, it's likely a persistent API issue
                    if consecutive_500_errors >= 10:
                        raise ValueError(
                            f"API appears to have persistent server issues (10+ consecutive 500 errors). "
                            f"This may be a problem with the {self.get_parameter_value('model')} model endpoint. "
                            f"Try using flux-kontext-pro instead, or wait and retry later."
                        )
                    continue
                    
                elif response.status_code == 429:
                    self.append_value_to_parameter(
                        "status", f"Attempt {attempt}: Rate limited, waiting longer...\n"
                    )
                    time.sleep(5)  # Additional wait for rate limiting
                    continue
                    
                elif response.status_code != 200:
                    response.raise_for_status()

                # Reset consecutive error counter on successful response
                consecutive_500_errors = 0
                
                result = response.json()
                status = result.get("status")

                self.append_value_to_parameter(
                    "status", f"Attempt {attempt}: {status}\n"
                )

                if status == "Ready":
                    image_url = result.get("result", {}).get("sample")
                    if not image_url:
                        raise ValueError(f"No image URL in result: {result}")
                    
                    # Extract the actual seed used by the API
                    api_seed = result.get("result", {}).get("seed")
                    if api_seed:
                        self.append_value_to_parameter("status", f"API used seed: {api_seed}\n")
                    return image_url, api_seed

                elif status in ["Processing", "Queued", "Pending"]:
                    continue

                else:
                    raise ValueError(
                        f"Generation failed with status '{status}': {result}"
                    )

            except requests.RequestException as e:
                # Only log non-500 errors here since we handle 500s above
                if not (hasattr(e, 'response') and e.response and e.response.status_code == 500):
                    self.append_value_to_parameter(
                        "status", f"Request error on attempt {attempt}: {str(e)}\n"
                    )
                if attempt >= max_attempts:
                    raise

        raise TimeoutError(f"Generation timed out after {max_attempts} attempts")

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

            return ImageUrlArtifact(
                value=static_url, name=f"bfl_{model}_{seed_str}_{timestamp}"
            )
        except Exception as e:
            raise ValueError(f"Failed to create image artifact: {str(e)}")

    def _poll_and_process_result(self, api_key: str, request_id: str, output_format: str) -> None:
        """Poll for result, download image, and set output - called via yield."""
        try:
            # Poll for result
            image_url, api_seed = self._poll_for_result(api_key, request_id)
            
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

    def process(self) -> AsyncResult[None]:
        """Non-blocking entry point for Griptape engine."""
        yield lambda: self._process()

    def _process(self) -> None:
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
