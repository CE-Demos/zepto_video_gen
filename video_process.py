import os
from google.cloud import storage
import cv2 # OpenCV for video processing
from moviepy.editor import VideoFileClip, concatenate_videoclips
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import time
import os # Make sure os is imported for file operations
from vertexai.generative_models import GenerativeModel, Image
import tempfile

# --- Configuration ---

GCS_BUCKET_NAME = "veo_exps"  # Replace with your bucket name
VIDEO_SOURCE_PATH = "input_videos/studio_bg.mp4"  # Path to video in GCS
FRAME_EXTRACTION_TIME_SECONDS = 7  # Time in seconds to extract frame
BACKGROUND_IMAGE_FOLDER = "background_output/"
BACKGROUND_IMAGE_NAME = "background.jpg"
PRODUCT_IMAGES_FOLDER = "images_for_gen/"
STUDIO_PRODUCTS_FOLDER = "studio_products/"
STUDIO_VIDEOS_FOLDER = "studio_videos/"
FINAL_VIDEOS_FOLDER = "final_videos/"
FINAL_VIDEO_NAME = "final_product_showcase.mp4"

TEMP_LOCAL_FILES_DIR = "temp_image_files" # Define a directory for temporary files
os.makedirs(TEMP_LOCAL_FILES_DIR, exist_ok=True) 

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

def imagen3_remove_person_and_get_background(image_bytes: bytes) -> bytes | None:
    """
    Placeholder for Imagen3 API call to remove any person from an image and return only the background.
    This is an IMAGE EDITING task (likely inpainting or object removal).

    Args:
        image_bytes: Bytes of the input image.
        project_id: Your Google Cloud Project ID.
        location: The GCP region where your Imagen model/endpoint is.
    Returns:
        Bytes of the background image, or None if failed.
    """
    print(f"INFO: (Placeholder) Calling Imagen3 (model for image editing/inpainting) "
        f"to remove person/get background from image.")

    
    #from google.cloud import aiplatform # Or the specific Imagen SDK if different
    # from vertexai.preview.generative_models import GenerativeModel, Part # Example for Gemini, Imagen might differ
    
    try:
      # Initialize client, model, etc. based on official Imagen SDK for Vertex AI
      # This might involve specifying an endpoint for an editing model.
        client = genai.Client(api_key='AIzaSyCwtqZvVOiEx86-ZY1Xssn1sw6sikVLia0')
    
      # The model name or endpoint ID for Imagen3 image editing/inpainting
      # This is NOT the same as text-to-image generation model.
       # imagen_editing_model_name = "imagen-3.0-capability-001" # THIS IS A HYPOTHETICAL NAME
    
      # Construct the request for Imagen3.
      # This will depend on the specific API for inpainting or object removal.
      # It will likely take the image_bytes and a prompt or instruction.
      # For example:
        # edit_prompt = "Remove all people from the image. Retain only the background. Fill in the removed areas naturally."
    
        client = genai.Client(api_key='AIzaSyCwtqZvVOiEx86-ZY1Xssn1sw6sikVLia0')

        from vertexai.preview.vision_models import (
            Image,
            ImageGenerationModel,
            ControlReferenceImage,
            StyleReferenceImage,
            SubjectReferenceImage,
            RawReferenceImage,
        )

        generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")

        reference_images = [
            SubjectReferenceImage(
                reference_id=1,
                image=image_bytes,  # Added comma here
                subject_type="SUBJECT_TYPE_PERSON",
            ),  # This comma separates the two SubjectReferenceImage objects in the list
        ]

        response = generation_model._generate_images(
            prompt="Remove all people from the image [1]. Retain only the background of the studio with the camera and light stands intact.",
            number_of_images=1,
            negative_prompt="",
            aspect_ratio="9:16",
            person_generation="allow_adult",
            safety_filter_level="block_few",
            reference_images=reference_images,
        )

        if not response: # Or check response.images or similar if the structure is different
            print("ERROR: Imagen3 API returned an empty response.")
            return None
        
        generated_image_object = response[0]

        
        # If this fails, try '._image_bytes' or consult the specific documentation for the type of 'generated_image_object'.
        if hasattr(generated_image_object, 'image_bytes'):
            processed_background_bytes = generated_image_object.image_bytes
            print(f"INFO: Successfully extracted {len(processed_background_bytes)} bytes from Imagen3 response object.")
            return processed_background_bytes
        elif hasattr(generated_image_object, '_image_bytes'): # Fallback for some SDK versions/objects
            processed_background_bytes = generated_image_object._image_bytes
            print(f"INFO: Successfully extracted {len(processed_background_bytes)} bytes using ._image_bytes from Imagen3 response object.")
            return processed_background_bytes
        else:
            print(f"ERROR: Could not find a '.image_bytes' or '._image_bytes' attribute on the response object: {type(generated_image_object)}")
            print(f"       Attributes available: {dir(generated_image_object)}")
            return None
    
    #     print("INFO: (Placeholder) Successfully called hypothetical Imagen3 edit API.")
    #   # For testing the pipeline flow, return the original image:
    #     # background_image_bytes = image_bytes
    #     return background_image_bytes
    
    except Exception as e:
      print(f"ERROR: (Placeholder) Actual Imagen3 API call for person removal failed: {e}")
      print("       Please implement this function using the correct Imagen SDK and model for IMAGE EDITING/INPAINTING.")
      return None


    print("WARN: (Placeholder) imagen3_remove_person_and_get_background returning original image as placeholder.")
    return image_bytes

def imagen3_replace_background(product_image_bytes: bytes, background_image_bytes: bytes) -> bytes | None:
    """
    Placeholder for Imagen3 API call to replace the background of a product image
    with a new background image. This is an IMAGE EDITING or COMPOSITING task.

    Args:
        product_image_bytes: Bytes of the product image (foreground).
        background_image_bytes: Bytes of the new background image.
        project_id: Your Google Cloud Project ID.
        location: The GCP region where your Imagen model/endpoint is.
    Returns:
        Bytes of the product image with the new background, or None if failed.
    """
    print(f"INFO: (Placeholder) Calling Imagen3 (model for image editing/compositing) "
        f"to replace background for images.")

    # --- Replace with actual Imagen3 SDK call for background replacement ---
    # This will require an Imagen3 model/API capable of image compositing or
    # background replacement, taking a foreground image and a background image.
    #
    # from google.cloud import aiplatform # Or the specific Imagen SDK
    try:
        client = genai.Client(api_key='AIzaSyCwtqZvVOiEx86-ZY1Xssn1sw6sikVLia0')

        from vertexai.preview.vision_models import (
            Image,
            ImageGenerationModel,
            ControlReferenceImage,
            StyleReferenceImage,
            SubjectReferenceImage,
            RawReferenceImage,
        )

        generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")

        reference_images = [
            SubjectReferenceImage(
                reference_id=1,
                image=product_image_bytes,  # Added comma here
                subject_type="SUBJECT_TYPE_PERSON",
            ),  # This comma separates the two SubjectReferenceImage objects in the list
            SubjectReferenceImage(
                reference_id=2,
                image=background_image_bytes,  # Added comma here
                subject_description="",        # Added comma here
                subject_type="SUBJECT_TYPE_DEFAULT",
            ),  # This trailing comma for the list itself is optional but good practice
        ]

        response = generation_model._generate_images(
            prompt="Add image [2] as background for image [1]. Keep the light sounds and camera in the studio background from image [2] intact ",
            number_of_images=1,
            negative_prompt="",
            aspect_ratio="9:16",
            person_generation="allow_adult",
            safety_filter_level="block_few",
            reference_images=reference_images,
        )

        if not response: # Or check response.images or similar if the structure is different
            print("ERROR: Imagen3 API returned an empty response.")
            return None
        
        generated_image_object = response[0]

        if hasattr(generated_image_object, 'image_bytes'):
            processed_background_bytes = generated_image_object.image_bytes
            print(f"INFO: Successfully extracted {len(processed_background_bytes)} bytes from Imagen3 response object.")
            return processed_background_bytes
        elif hasattr(generated_image_object, '_image_bytes'): # Fallback for some SDK versions/objects
            processed_background_bytes = generated_image_object._image_bytes
            print(f"INFO: Successfully extracted {len(processed_background_bytes)} bytes using ._image_bytes from Imagen3 response object.")
            return processed_background_bytes
        else:
            print(f"ERROR: Could not find a '.image_bytes' or '._image_bytes' attribute on the response object: {type(generated_image_object)}")
            print(f"       Attributes available: {dir(generated_image_object)}")
            return None

        # Adjust based on actual response
    #
    #     print("INFO: (Placeholder) Successfully called hypothetical Imagen3 background replacement API.")
    # # For testing the pipeline flow, return the original product image:
    #     # studio_product_bytes = product_image_bytes
    #     return studio_product_bytes
    # #
    except Exception as e:
        print(f"ERROR: (Placeholder) Actual Imagen3 API call for background replacement failed: {e}")
        print("       Please implement this function using the correct Imagen SDK and model for IMAGE EDITING/COMPOSITING.")
        return None

    # Fallback placeholder behavior:
    # print("WARN: (Placeholder) imagen3_replace_background returning original product image as placeholder.")
    # return product_image_bytes

def veo2_generate_video_from_image(image_bytes: bytes, image_mime_type: str, prompt: str) -> bytes | None:
    """
    Placeholder for Veo2 API call to generate video from an image and prompt.

    Args:
        image_bytes: Bytes of the input image.
        prompt: Text prompt to control video generation.
        project_id: Your Google Cloud Project ID.
        location: The GCP region where your Veo2 model/endpoint is.
    Returns:
        Bytes of the generated video, or None if failed.
    """
    print(f"INFO: (Placeholder) Calling Veo2 (model for image-to-video generation) "
        f"for studio images with prompt: '{prompt}'...")

    # --- Replace with actual Veo2 SDK call ---
    # Example (consult official Veo2/Vertex AI documentation):
    # from google.cloud import aiplatform # Or specific Veo2 SDK
   # temp_image_file_path = None 

    image_input_for_sdk = {
            "mime_type": image_mime_type,
            "data": image_bytes  # Raw bytes are often accepted here by the SDK
        }


    try:
    #   # Initialize client, model, etc.
       client = genai.Client(api_key='AIzaSyCwtqZvVOiEx86-ZY1Xssn1sw6sikVLia0')

       veo_model_name = "veo-2.0-generate-001" # THIS IS A HYPOTHETICAL NAME

    #    extension = ".bin" # Default extension
    #    if image_mime_type == "image/png":
    #         extension = ".png"
    #    elif image_mime_type in ["image/jpeg", "image/jpg"]:
    #         extension = ".jpg"

    #    with tempfile.NamedTemporaryFile(delete=False, suffix=extension, mode='wb') as tmp_file:
    #         tmp_file.write(image_bytes)
    #         temp_image_file_path = tmp_file.name
    #    print(f"INFO: Input image bytes temporarily saved to: {temp_image_file_path}")
    #
       response =client.models.generate_videos( # or veo_client.generate_from_image(...)
            model=veo_model_name,
            image=image_input_for_sdk,
            prompt=prompt,
            config=types.GenerateVideosConfig(
      
                aspect_ratio="9:16",  
                number_of_videos=1
            ),
            
        )
       
       if not response: # Or check response.videos or similar if the structure is different
            print("ERROR: Veo2 API (via genai.Client) returned an empty response.")
            return None
       
       generated_video_object = response[0]

       video_byte_data = None
       if hasattr(generated_video_object, 'video_bytes'):
            video_byte_data = generated_video_object.video_bytes
       elif hasattr(generated_video_object, 'media_bytes'): # Another common possibility
            video_byte_data = generated_video_object.media_bytes
       elif hasattr(generated_video_object, 'content'): # Generic attribute
            video_byte_data = generated_video_object.content
    #
       if video_byte_data:
            print(f"INFO: Successfully extracted {len(video_byte_data)} bytes from Veo2 response object.")
            return video_byte_data
       else:
            print(f"ERROR: Could not find a suitable bytes attribute (e.g., '.video_bytes', '.media_bytes', '.content') on the response object: {type(generated_video_object)}")
            print(f"       Attributes available: {dir(generated_video_object)}")
            return None
    #
    except Exception as e:
       print(f"ERROR: (Placeholder) Actual Veo2 API call failed: {e}")
       print("       Please implement this function using the correct Veo2/Vertex AI SDK.")
       return None

    # Fallback placeholder behavior:
    print("WARN: (Placeholder) veo2_generate_video_from_image returning dummy video bytes as placeholder.")
    return f"dummy_video_content_for_image_and_prompt_{prompt.replace(' ', '_')}_{time.time()}.mp4".encode('utf-8')

# def get_bytes_from_image_file(file_path: str) -> bytes | None:
#     """Reads an image file and returns its content as bytes."""
#     try:
#         with open(file_path, "rb") as image_file:  # "rb" means read in binary mode
#             image_bytes = image_file.read()
#         print(f"Successfully read {len(image_bytes)} bytes from {file_path}")
#         return image_bytes
#     except FileNotFoundError:
#         print(f"Error: The file '{file_path}' was not found.")
#         return None
#     except Exception as e:
#         print(f"Error reading file '{file_path}': {e}")
#         return None

# --- Helper Functions for GCS ---

def download_blob_to_memory(blob_name: str) -> bytes:
    """Downloads a blob from GCS into memory."""
    blob = bucket.blob(blob_name)
    print(f"INFO: Downloading gs://{GCS_BUCKET_NAME}/{blob_name} to memory...")
    content = blob.download_as_bytes()
    print(f"INFO: Downloaded {len(content)} bytes.")
    return content

def download_blob_to_file(blob_name: str, destination_file_name: str):
    """Downloads a blob from GCS to a local file."""
    blob = bucket.blob(blob_name)
    print(f"INFO: Downloading gs://{GCS_BUCKET_NAME}/{blob_name} to {destination_file_name}...")
    blob.download_to_filename(destination_file_name)
    print(f"INFO: Download complete.")

def upload_blob_from_memory(destination_blob_name: str, data: bytes, content_type='image/jpeg'):
    """Uploads data from memory to GCS."""
    blob = bucket.blob(destination_blob_name)
    print(f"INFO: Uploading data to gs://{GCS_BUCKET_NAME}/{destination_blob_name}...")
    blob.upload_from_string(data, content_type=content_type)
    print(f"INFO: Upload complete.")

def upload_blob_from_file(source_file_name: str, destination_blob_name: str):
    """Uploads a file to GCS."""
    blob = bucket.blob(destination_blob_name)
    print(f"INFO: Uploading {source_file_name} to gs://{GCS_BUCKET_NAME}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_name)
    print(f"INFO: Upload complete.")

def list_blobs_in_folder(folder_prefix: str):
    """Lists all blobs in a GCS folder."""
    print(f"INFO: Listing blobs in gs://{GCS_BUCKET_NAME}/{folder_prefix}...")
    blobs = storage_client.list_blobs(GCS_BUCKET_NAME, prefix=folder_prefix)
    return [blob for blob in blobs if not blob.name.endswith('/')] # Exclude folder itself


# --- Main Program Steps ---

def step_1_extract_frame_from_video():
    """
    Takes a video from GCS, extracts a frame, and returns its bytes.
    """
    print("\n--- Step 1: Extracting Frame from Video ---")
    local_video_path = "temp_video.mp4"
    download_blob_to_file(VIDEO_SOURCE_PATH, local_video_path)

    vidcap = cv2.VideoCapture(local_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * FRAME_EXTRACTION_TIME_SECONDS)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image_cv2 = vidcap.read()

    if success:
        print(f"INFO: Successfully extracted frame {frame_number} from {VIDEO_SOURCE_PATH}")
        _, image_bytes = cv2.imencode('.jpg', image_cv2)
        os.remove(local_video_path) # Clean up local file
        return image_bytes.tobytes()
    else:
        print(f"ERROR: Could not extract frame from {VIDEO_SOURCE_PATH} at {FRAME_EXTRACTION_TIME_SECONDS}s.")
        os.remove(local_video_path)
        return None

def step_2_generate_and_store_background(frame_bytes: bytes):
    """
    Uses Imagen3 to remove person from the frame and stores it as background.
    """
    print("\n--- Step 2: Generating and Storing Background Image ---")
    if not frame_bytes:
        print("ERROR: No frame image provided to generate background.")
        return None

    background_image_bytes = imagen3_remove_person_and_get_background(frame_bytes)

    if background_image_bytes:
        destination_blob_name = os.path.join(BACKGROUND_IMAGE_FOLDER, BACKGROUND_IMAGE_NAME)
        upload_blob_from_memory(destination_blob_name, background_image_bytes, content_type='image/jpeg')
        print(f"INFO: Background image stored at gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
        return background_image_bytes
    else:
        print("ERROR: Failed to generate background image using Imagen3.")
        return None

def step_3_replace_background_for_products(background_image_bytes: bytes):
    """
    Replaces background for product images using the generated background.
    """
    print("\n--- Step 3: Replacing Background for Product Images ---")
    if not background_image_bytes:
        print("ERROR: No background image provided for product background replacement.")
        return

    product_image_blobs = list_blobs_in_folder(PRODUCT_IMAGES_FOLDER)
    if not product_image_blobs:
        print(f"INFO: No product images found in gs://{GCS_BUCKET_NAME}/{PRODUCT_IMAGES_FOLDER}")
        return

    print(f"INFO: Found {len(product_image_blobs)} product images.")
    for blob in product_image_blobs:
        if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"INFO: Processing product image: {blob.name}")
            product_image_bytes = download_blob_to_memory(blob.name)
            studio_product_bytes = imagen3_replace_background(product_image_bytes, background_image_bytes)

            if studio_product_bytes:
                base_filename = os.path.basename(blob.name)
                destination_blob_name = os.path.join(STUDIO_PRODUCTS_FOLDER, base_filename)
                upload_blob_from_memory(destination_blob_name, studio_product_bytes, content_type=blob.content_type)
                print(f"INFO: Studio product image stored at gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
            else:
                print(f"ERROR: Failed to replace background for {blob.name}")
        else:
            print(f"INFO: Skipping non-image file: {blob.name}")


def step_4_generate_videos_from_studio_images(user_prompt: str):
    """
    Generates videos from studio product images using Veo2 and a user prompt.
    """
    print("\n--- Step 4: Generating Videos from Studio Product Images ---")
    studio_image_blobs = list_blobs_in_folder(STUDIO_PRODUCTS_FOLDER)

    if not studio_image_blobs:
        print(f"INFO: No studio product images found in gs://{GCS_BUCKET_NAME}/{STUDIO_PRODUCTS_FOLDER}")
        return

    print(f"INFO: Found {len(studio_image_blobs)} studio product images.")
    for blob in studio_image_blobs:
        if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"INFO: Generating video for: {blob.name} with prompt: '{user_prompt}'")
            image_bytes = download_blob_to_memory(blob.name)
            image_mime_type="image/png"
            video_bytes = veo2_generate_video_from_image(image_bytes, image_mime_type, user_prompt)

            if video_bytes:
                base_filename = os.path.splitext(os.path.basename(blob.name))[0] + ".mp4"
                destination_blob_name = os.path.join(STUDIO_VIDEOS_FOLDER, base_filename)
                # Assuming Veo2 returns MP4 bytes. Adjust content_type if needed.
                upload_blob_from_memory(destination_blob_name, video_bytes, content_type='video/mp4')
                print(f"INFO: Studio video stored at gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
            else:
                print(f"ERROR: Failed to generate video for {blob.name}")
        else:
            print(f"INFO: Skipping non-image file for video generation: {blob.name}")


def step_5_concatenate_videos_and_upload():
    """
    Concatenates all videos from the studio_videos folder and uploads the final video.
    """
    print("\n--- Step 5: Concatenating Videos and Uploading Final Video ---")
    video_blobs = list_blobs_in_folder(STUDIO_VIDEOS_FOLDER)

    if not video_blobs:
        print(f"INFO: No videos found in gs://{GCS_BUCKET_NAME}/{STUDIO_VIDEOS_FOLDER} to concatenate.")
        return

    local_video_files = []
    temp_dir = "temp_videos_for_concat"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"INFO: Downloading {len(video_blobs)} videos for concatenation...")
    for i, blob in enumerate(video_blobs):
        if blob.name.lower().endswith('.mp4'): # Assuming MP4 format from Veo2
            local_file_path = os.path.join(temp_dir, f"video_{i}_{os.path.basename(blob.name)}")
            download_blob_to_file(blob.name, local_file_path)
            local_video_files.append(local_file_path)
        else:
            print(f"INFO: Skipping non-mp4 file: {blob.name}")


    if not local_video_files:
        print("INFO: No valid video files downloaded for concatenation.")
        if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
        return

    try:
        print("INFO: Concatenating videos using MoviePy...")
        video_clips = [VideoFileClip(file_path) for file_path in local_video_files]
        final_clip = concatenate_videoclips(video_clips, method="compose")
        final_video_local_path = os.path.join(temp_dir, FINAL_VIDEO_NAME)
        final_clip.write_videofile(final_video_local_path, codec="libx264", audio_codec="aac") # Or other suitable codecs

        print(f"INFO: Concatenation complete. Final video at {final_video_local_path}")

        final_video_gcs_path = os.path.join(FINAL_VIDEOS_FOLDER, FINAL_VIDEO_NAME)
        upload_blob_from_file(final_video_local_path, final_video_gcs_path)
        print(f"INFO: Final video uploaded to gs://{GCS_BUCKET_NAME}/{final_video_gcs_path}")

    except Exception as e:
        print(f"ERROR: An error occurred during video concatenation or upload: {e}")
    finally:
        # Clean up local temporary files and directory
        print("INFO: Cleaning up temporary local files...")
        for clip in video_clips:
            clip.close() # Important to close clips
        for file_path in local_video_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(final_video_local_path) and os.path.isfile(final_video_local_path):
             os.remove(final_video_local_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        print("INFO: Cleanup complete.")


def main():
    """
    Main function to run all steps in sequence.
    """
    print("Starting Product Video Generation Pipeline...")

    # Step 1: Extract frame
    extracted_frame_bytes = step_1_extract_frame_from_video()

    # Step 2: Generate background
    if extracted_frame_bytes:
        background_bytes = step_2_generate_and_store_background(extracted_frame_bytes)
    else:
        print("ERROR: Halting pipeline due to failure in Step 1.")
        return

    # Step 3: Replace background for products
    if background_bytes:
        step_3_replace_background_for_products(background_bytes)
    else:
        print("ERROR: Halting pipeline due to failure in Step 2.")
        return

    # Step 4: Generate videos from studio images
    # You might want to make this prompt dynamic (e.g., from config file or user input)
    video_generation_prompt = input("Enter the text prompt for video generation (e.g., 'smooth panning shot'): ")
    if not video_generation_prompt:
        video_generation_prompt = "a gentle zoom in on the product" # Default prompt
        print(f"INFO: No prompt entered, using default: '{video_generation_prompt}'")

    step_4_generate_videos_from_studio_images(video_generation_prompt)

    # Step 5: Concatenate videos and upload
    step_5_concatenate_videos_and_upload()

    print("\nProduct Video Generation Pipeline Finished.")

if __name__ == "__main__":
    # --- Prerequisites ---
    # 1. Google Cloud SDK authenticated (e.g., via `gcloud auth application-default login`)
    # 2. `GOOGLE_APPLICATION_CREDENTIALS` environment variable might be needed depending on your auth setup.
    # 3. The GCS bucket and source folders/video must exist.
    # 4. Placeholder functions for Imagen3 and Veo2 need to be implemented with actual API calls.

    # --- Create necessary folders in GCS if they don't exist (optional, for setup) ---
    # You can do this via gsutil or the Cloud Console.
    # Example:
    # gsutil mkdir gs://your-gcs-bucket-name/source_videos/
    # gsutil mkdir gs://your-gcs-bucket-name/background_output/
    # gsutil mkdir gs://your-gcs-bucket-name/product_images_input/
    # gsutil mkdir gs://your-gcs-bucket-name/studio_products/
    # gsutil mkdir gs://your-gcs-bucket-name/studio_videos/
    # gsutil mkdir gs://your-gcs-bucket-name/final_videos/
    # And upload your initial video and product images.

    main()