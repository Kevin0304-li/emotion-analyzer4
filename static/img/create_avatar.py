from PIL import Image, ImageDraw, ImageFont
import os

def create_user_avatar(save_path, size=200, bg_color="#6c5ce7", text_color="#ffffff"):
    """
    Create a simple user avatar with the letter 'U' in the center
    """
    # Create a new image with the given background color
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=int(size * 0.6))
    except IOError:
        font = ImageFont.load_default()
    
    # Draw the letter 'U' in the center
    text = "U"
    text_width, text_height = draw.textsize(text, font=font)
    position = ((size - text_width) // 2, (size - text_height) // 2)
    draw.text(position, text, fill=text_color, font=font)
    
    # Save the image
    img.save(save_path)
    print(f"Avatar created and saved to {save_path}")

if __name__ == "__main__":
    # Create the avatar in the current directory
    avatar_path = os.path.join(os.path.dirname(__file__), "user-avatar.png")
    create_user_avatar(avatar_path) 