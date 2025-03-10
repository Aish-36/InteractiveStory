import streamlit as st
import os
import cv2
import tempfile
import numpy as np
from gtts import gTTS
from pathlib import Path
from PIL import Image, ImageSequence
import logging
# Remove imageio import
import time
import shutil
import subprocess
from deep_translator import GoogleTranslator

class StorySignConverter:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        self.base_dir = Path.cwd()
        self.videos_dir = self.base_dir / "videos"
        self.temp_dir = self.base_dir / "temp"
        
        # Create directories if they don't exist
        for directory in [self.videos_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
            self.logger.info(f"Directory created/verified at: {directory.absolute()}")
        
        # Initialize app
        self.setup_page()
        self.init_session_state()
        self.load_resources()
        self.run()

    def setup_page(self):
        """Setup page configuration and styling"""
        st.set_page_config(page_title="Interactive Story Signing", layout="wide")
        
        # Initialize theme if not in session state
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
        
        # Apply CSS styling
        self.apply_styling()

    def apply_styling(self):
        """Apply CSS styling to the app"""
        theme_css = self.get_theme_css()
        st.markdown(f"""
            <style>
                {theme_css}
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
                
                .stApp {{
                    font-family: 'Poppins', sans-serif;
                    background-color: var(--background-color);
                }}
                
                .main-title {{
                    text-align: center;
                    padding: 2rem;
                    font-size: 2.5rem;
                    font-weight: 600;
                    margin-bottom: 2rem;
                    color: var(--text-color);
                    background: linear-gradient(45deg, var(--gradient-start), var(--gradient-end));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                
                .story-container {{
                    background-color: var(--card-bg);
                    padding: 2rem;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin: 2rem 0;
                }}
                
                .controls-container {{
                    display: flex;
                    justify-content: center;
                    gap: 2rem;
                    margin: 2rem 0;
                }}
                
                .custom-button {{
                    background-color: var(--button-bg);
                    color: var(--button-text);
                    padding: 0.75rem 1.5rem;
                    border-radius: 8px;
                    border: none;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    font-weight: 600;
                }}
                
                .custom-button:hover {{
                    transform: scale(1.05);
                    background-color: var(--hover-color);
                }}
                
                .status-message {{
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                    background-color: var(--status-bg);
                    color: var(--status-text);
                }}

            .download-button {{
                background-color: var(--button-bg);
                color: var(--button-text);
                padding: 1rem 2rem;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 600;
                display: block;
                margin: 2rem auto;
                text-align: center;
            }}

            .download-button:hover {{
                transform: scale(1.05);
                background-color: var(--hover-color);
            }}
            </style>
        """, unsafe_allow_html=True)

    def get_theme_css(self):
        """Get theme-specific CSS variables"""
        if st.session_state.theme == 'dark':
            return """
                :root {
                    --background-color: #1a1a1a;
                    --text-color: #ffffff;
                    --card-bg: #2d2d2d;
                    --button-bg: #4CAF50;
                    --button-text: #ffffff;
                    --hover-color: #45a049;
                    --gradient-start: #4CAF50;
                    --gradient-end: #45a049;
                    --status-bg: #2d2d2d;
                    --status-text: #ffffff;
                }
            """
        else:
            return """
                :root {
                    --background-color: #f5f7fa;
                    --text-color: #2c3e50;
                    --card-bg: #ffffff;
                    --button-bg: #4CAF50;
                    --button-text: #ffffff;
                    --hover-color: #45a049;
                    --gradient-start: #2c3e50;
                    --gradient-end: #3498db;
                    --status-bg: #e8f5e9;
                    --status-text: #2c3e50;
                }
            """

    def init_session_state(self):
        """Initialize session state variables"""
        if 'current_story' not in st.session_state:
            st.session_state.current_story = ""
        if 'story_signs' not in st.session_state:
            st.session_state.story_signs = []
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        if 'audio_files' not in st.session_state:
            st.session_state.audio_files = []
        if 'video_paths' not in st.session_state:
            st.session_state.video_paths = []
        if 'selected_language' not in st.session_state:
            st.session_state.selected_language = 'en'
    def load_resources(self):
        """Load image and GIF resources"""
        self.resources = {}
        
        # Define paths - update these to your actual paths
        image_dir = Path("Images")  # Directory for static images
        gif_dir = Path("Gifs")      # Directory for GIFs
        
        # Load static images
        if image_dir.exists():
            for ext in ['.jpeg', '.jpg', '.png']:
                for img_path in image_dir.glob(f"*{ext}"):
                    key = img_path.stem.upper()
                    self.resources[key] = {
                        'path': str(img_path),
                        'type': 'image'
                    }
                    self.logger.info(f"Loaded image: {key}")

        # Load GIFs
        if gif_dir.exists():
            for gif_path in gif_dir.glob("*.gif"):
                key = gif_path.stem.upper()
                self.resources[key] = {
                    'path': str(gif_path),
                    'type': 'gif'
                }
                self.logger.info(f"Loaded GIF: {key}")

    def process_gif(self, gif_path, target_size=(640, 480)):
        """Process GIF and return list of frames"""
        frames = []
        with Image.open(gif_path) as gif:
            for frame in ImageSequence.Iterator(gif):
                # Convert frame to RGB
                frame = frame.convert('RGB')
                # Resize frame
                frame = frame.resize(target_size, Image.Resampling.LANCZOS)
                # Convert to numpy array
                frame_array = np.array(frame)
                frames.append(frame_array)
        return frames
    
    def create_video(self, words_list, sentence_index):
        """Create video from list of words, where each word's images are shown in one frame"""
        if not words_list:
            return None

        FRAME_HEIGHT = 480
        FRAME_WIDTH = 1280  # Wider frame to accommodate multiple images
        BASE_IMAGE_WIDTH = 160  # Base individual image width
        BASE_IMAGE_HEIGHT = 240  # Base individual image height
        FRAME_RATE = 24
        STATIC_DURATION = 2  # seconds to show static images
        
        # Create unique video filename
        video_filename = f"sentence_{sentence_index}_{int(time.time())}.mp4"
        output_path = self.videos_dir / video_filename

        all_frames = []

        for word in words_list:
            # Create a blank frame for this word
            word_frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255  # White background

            # Get all characters for this word
            chars = [c for c in word.upper() if c in self.resources]
            if not chars:
                continue

            try:
                num_images = len(chars)
                space_between = 20  # pixels between images
                max_allowed_width = FRAME_WIDTH - 40  # 20 pixels margin on each side

                # Adjust image width if word is too long to fit in one row
                if num_images * BASE_IMAGE_WIDTH + (num_images - 1) * space_between > max_allowed_width:
                    IMAGE_WIDTH = max_allowed_width // num_images
                    space_between = (max_allowed_width - num_images * IMAGE_WIDTH) // (num_images - 1) if num_images > 1 else 0
                else:
                    IMAGE_WIDTH = BASE_IMAGE_WIDTH

                IMAGE_HEIGHT = int(IMAGE_WIDTH * 1.5)  # Maintain aspect ratio
                max_chars_per_line = FRAME_WIDTH // (IMAGE_WIDTH + space_between)

                # Handle multi-line words
                if num_images > max_chars_per_line:
                    lines = [chars[i:i + max_chars_per_line] for i in range(0, num_images, max_chars_per_line)]
                else:
                    lines = [chars]

                start_y = (FRAME_HEIGHT - (len(lines) * IMAGE_HEIGHT + (len(lines) - 1) * 20)) // 2  # Center vertically

                # Process each line of characters
                for line_index, line in enumerate(lines):
                    total_width = len(line) * IMAGE_WIDTH + (len(line) - 1) * space_between
                    start_x = (FRAME_WIDTH - total_width) // 2
                    line_start_y = start_y + (IMAGE_HEIGHT + 20) * line_index  # Adjust vertical placement

                    for i, char in enumerate(line):
                        resource_info = self.resources.get(char)
                        if not resource_info:
                            continue

                        x_pos = start_x + i * (IMAGE_WIDTH + space_between)

                        try:
                            # Load and resize image
                            img = Image.open(resource_info['path']).convert('RGB')
                            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
                            img_array = np.array(img)

                            # Place image on correct row
                            word_frame[line_start_y:line_start_y + IMAGE_HEIGHT, x_pos:x_pos + IMAGE_WIDTH] = img_array

                        except Exception as e:
                            self.logger.error(f"Error processing character {char}: {str(e)}")
                            continue

                # Add the frame multiple times for duration
                for _ in range(int(FRAME_RATE * STATIC_DURATION)):
                    all_frames.append(word_frame.copy())

            except Exception as e:
                self.logger.error(f"Error processing word {word}: {str(e)}")
                continue

        # If we have frames, create the video
        if not all_frames:
            self.logger.error("No frames were generated")
            return None

        self.logger.info(f"Generated {len(all_frames)} frames")
        self.logger.info(f"Frame shape: {all_frames[0].shape}")
        self.logger.info(f"Frame dtype: {all_frames[0].dtype}")

        try:
            # Use OpenCV to write video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                FRAME_RATE,
                (FRAME_WIDTH, FRAME_HEIGHT),
                isColor=True
            )

            if not out.isOpened():
                self.logger.error("Failed to open video writer")
                return None

            try:
                for frame in all_frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()
                self.logger.info(f"Successfully created video at {output_path}")
                return str(output_path)

            except Exception as e:
                out.release()
                self.logger.error(f"Error writing video frames: {str(e)}")
                if output_path.exists():
                    Path(output_path).unlink()
                return None

        except Exception as e:
            self.logger.error(f"Error creating video: {str(e)}")
            return None
        
        

    def generate_audio(self, text):
        """Generate audio file from text"""
        try:
            tts = gTTS(text=text, lang='en')
            audio_path = self.temp_dir / f"audio_{int(time.time())}.mp3"
            tts.save(str(audio_path))
            return str(audio_path)
        except Exception as e:
            self.logger.error(f"Error generating audio: {str(e)}")
            return None

    def process_story(self, text):
        
        """Process story text into signs, videos, and audio"""
      
            # Translate text to English if not already in English
        if st.session_state.selected_language != 'en':
            self.translator = GoogleTranslator(
                source=st.session_state.selected_language, 
                target='en'
            )
            text_for_signs = self.translator.translate(text)
            st.info(f"Translated text: {text_for_signs}")
            text = text_for_signs
        else:
            text_for_signs = text
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        video_paths = []
        audio_files = []
        
        for i, sentence in enumerate(sentences):
            # Generate audio
            audio_file = self.generate_audio(sentence)
            if audio_file:
                audio_files.append(audio_file)
            
            # Process sentence into words
            words = []
            for word in sentence.split():
                word = ''.join(c for c in word if c.isalnum())
                if any(c.upper() in self.resources for c in word):
                    words.append(word)
            
            # Create video
            video_path = self.create_video(words, i)
            if video_path:
                video_paths.append(video_path)
        
        st.session_state.video_paths = video_paths
        st.session_state.audio_files = audio_files
        st.session_state.current_index = 0
        
    def add_download_button(self):
        """Add a download button for the complete story"""
        if st.session_state.video_paths and st.session_state.audio_files:
            st.markdown('<div class="story-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("⬇️ Download Complete Story", key="download_story"):
                    with st.spinner("Preparing your story..."):
                        final_video = self.combine_videos_and_audio()
                        if final_video:
                            with open(final_video, 'rb') as f:
                                video_bytes = f.read()
                            st.download_button(
                                label="Download Story",
                                data=video_bytes,
                                file_name="complete_story.mp4",
                                mime="video/mp4"
                            )
                            st.success("Your story is ready for download!")
                        else:
                            st.error("Failed to prepare the story for download.")
            
            st.markdown('</div>', unsafe_allow_html=True)
   
    def display_video_with_audio(self):
        def convert_video(input_path, output_path):
            try:
                command = [
                    'ffmpeg',
                    '-y',  # Add -y flag for automatic overwrite
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'fast',
                    '-crf', '23',
                    output_path
                ]
                subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # Suppress prompts
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error converting video: {e}")
                return False

    
        try:
            video_path = st.session_state.video_paths[st.session_state.current_index]
            audio_path = st.session_state.audio_files[st.session_state.current_index]
            
            print(f"Attempting to load video from: {video_path}")
            print(f"Attempting to load audio from: {audio_path}")
            
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                print(f"Video file size: {file_size} bytes")
                
                converted_path = video_path.replace('.mp4', '_converted.mp4')
                
                # Remove existing converted file if it exists
                if os.path.exists(converted_path):
                    os.remove(converted_path)
                    
                if convert_video(video_path, converted_path):
                    with open(converted_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                    
                    # Clean up converted file after displaying
                    if os.path.exists(converted_path):
                        os.remove(converted_path)
            else:
                st.error(f"Video file not found at: {video_path}")
                
            if os.path.exists(audio_path):
                st.audio(audio_path)
            else:
                st.error(f"Audio file not found at: {audio_path}")
                
        except Exception as e:
            st.error(f"Error loading media: {str(e)}")
            print(f"Detailed error: {e}")
    def combine_videos_and_audio(self):
        try:
            temp_file_list = self.temp_dir / "file_list.txt"
            output_path = self.temp_dir / f"complete_story_{int(time.time())}.mp4"
            converted_paths = []

            # Convert each video to h264
            with open(temp_file_list, 'w') as f:
                for video_path in st.session_state.video_paths:
                    converted_path = Path(video_path).with_suffix('.converted.mp4')
                    subprocess.run([
                        'ffmpeg',
                        '-y',
                        '-i', video_path,
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        str(converted_path)
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    converted_paths.append(converted_path)
                    f.write(f"file '{converted_path}'\n")

            # Concatenate videos
            subprocess.run([
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(temp_file_list),
                '-c', 'copy',
                str(output_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Combine audio files
            combined_audio = self.temp_dir / "combined_audio.mp3"
            audio_list_file = self.temp_dir / "audio_list.txt"
            with open(audio_list_file, 'w') as f:
                for audio_path in st.session_state.audio_files:
                    f.write(f"file '{audio_path}'\n")

            subprocess.run([
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(audio_list_file),
                '-c', 'copy',
                str(combined_audio)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Final combination
            final_output = self.temp_dir / f"final_story_{int(time.time())}.mp4"
            subprocess.run([
                'ffmpeg',
                '-y',
                '-i', str(output_path),
                '-i', str(combined_audio),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                str(final_output)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Clean up converted files
            for path in converted_paths:
                if path.exists():
                    path.unlink()

            return str(final_output)

        except Exception as e:
            self.logger.error(f"Error combining videos and audio: {str(e)}")
            return None
    def display_navigation(self):
        """Display navigation controls"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("◀ Previous", 
                        disabled=st.session_state.current_index == 0,
                        key="prev_button"):
                st.session_state.current_index -= 1
        
        with col2:
            total = len(st.session_state.video_paths)
            st.markdown(
                f'<p style="text-align: center; font-size: 1.2rem;">Sentence {st.session_state.current_index + 1} of {total}</p>',
                unsafe_allow_html=True
            )
        
        with col3:
            if st.button("Next ▶", 
                        disabled=st.session_state.current_index == len(st.session_state.video_paths) - 1,
                        key="next_button"):
                st.session_state.current_index += 1

    def run(self):
        """Main application loop"""
        st.markdown('<h1 class="main-title">Interactive Story Signing</h1>', unsafe_allow_html=True)
        
        # Theme selection
        theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
        st.session_state.theme = theme.lower()
        
        # Language selection with expanded language support
        SUPPORTED_LANGUAGES = {
            'English': 'en',
            # Indian Languages
            'Hindi': 'hi',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Kannada': 'kn',
            'Malayalam': 'ml',
            'Marathi': 'mr',
            'Bengali': 'bn',
            'Gujarati': 'gu',
            'Punjabi': 'pa',
            'Urdu': 'ur',
            # Other Asian Languages
            'Chinese (Simplified)': 'zh-CN',
            'Chinese (Traditional)': 'zh-TW',
            'Japanese': 'ja',
            'Korean': 'ko',
            # European Languages
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            # Middle Eastern Languages
            'Arabic': 'ar',
            'Persian': 'fa',
            # Southeast Asian Languages
            'Thai': 'th',
            'Vietnamese': 'vi',
            'Indonesian': 'id',
            'Malay': 'ms'
        }
        
        # Group languages by region
        LANGUAGE_GROUPS = {
            'Indian Languages': ['Hindi', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Marathi', 'Bengali', 'Gujarati', 'Punjabi', 'Urdu'],
            'East Asian': ['Chinese (Simplified)', 'Chinese (Traditional)', 'Japanese', 'Korean'],
            'European': ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Russian'],
            'Middle Eastern': ['Arabic', 'Persian'],
            'Southeast Asian': ['Thai', 'Vietnamese', 'Indonesian', 'Malay']
        }

        # Language selection interface
        st.sidebar.markdown("### Language Settings")
        language_group = st.sidebar.selectbox(
            "Select Language Group",
            list(LANGUAGE_GROUPS.keys())
        )
        
        selected_lang = st.sidebar.selectbox(
            "Select Input Language",
            LANGUAGE_GROUPS[language_group],
            index=0
        )
        
        st.session_state.selected_language = SUPPORTED_LANGUAGES[selected_lang]
        
        # Language info display
        st.sidebar.markdown(f"""
        **Current Language**: {selected_lang}  
        **Language Code**: {st.session_state.selected_language}
        """)
        
        # Debug information in sidebar
        if st.sidebar.checkbox("Show Debug Information"):
            st.sidebar.write("### Loaded Resources:")
            st.sidebar.write(f"Total resources: {len(self.resources)}")
            st.sidebar.write("Available characters:", sorted(self.resources.keys()))
        
        # Story input section
        st.markdown('<div class="story-container">', unsafe_allow_html=True)
        
        # Default stories
        DEFAULT_STORIES = {
            "The Lion and the Mouse": "A Lion lay asleep in the forest. A tiny Mouse began running up and down upon him.",
            "The Fox and the Grapes": "One hot summer's day a Fox was strolling through an orchard.",
            "The Tortoise and the Hare": "A Hare was making fun of the Tortoise one day for being so slow."
        }
        
        input_method = st.radio("Choose input method:", ["Default Stories", "Custom Text"])
        
        if input_method == "Default Stories":
            story_title = st.selectbox("Select a story:", list(DEFAULT_STORIES.keys()))
            if st.button("Load Story", key="load_story"):
                with st.spinner("Processing story..."):
                    self.process_story(DEFAULT_STORIES[story_title])
                st.success("Story processed successfully!")
        else:
            text_input = st.text_area("Enter your text:", height=150)
            if st.button("Process Text", key="process_text") and text_input:
                with st.spinner("Processing text..."):
                    self.process_story(text_input)
                st.success("Text processed successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display section
        if st.session_state.video_paths:
            st.markdown('<div class="story-container">', unsafe_allow_html=True)
            self.display_video_with_audio()
            self.display_navigation()
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app = StorySignConverter()