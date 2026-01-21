import os
# Теперь этот импорт снова заработает
from moviepy.editor import VideoFileClip 
from cross_correlation import CorrelationTracker 

def create_demo_gif(video_path, output_name="demo.gif", duration=3):
    if not os.path.exists(video_path):
        print(f"❌ Ошибка: Не найден файл видео {video_path}")
        return

    print("🚀 Запуск трекера и генерация GIF...")
    
    tracker = CorrelationTracker(detection_rate=5)
    
    try:
        clip = VideoFileClip(video_path)
        # В версии 1.0.3 метод subclip гарантированно есть
        subclip = clip.subclip(0, duration)
        
        processed_clip = subclip.fl_image(tracker.update_frame)
        
        processed_clip.write_gif(output_name, fps=15)
        print(f"✅ Гифка сохранена: {output_name}")
        
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        
    VIDEO_SOURCE = os.path.join("data", "test.mp4") 
    
    if not os.path.exists(VIDEO_SOURCE):
        print(f"⚠️ Файл {VIDEO_SOURCE} не найден!")
    else:
        create_demo_gif(VIDEO_SOURCE, "tracking_demo.gif", duration=4)