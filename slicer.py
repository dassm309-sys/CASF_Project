from moviepy import VideoFileClip
import os
import glob

def slice_all_videos():
    # 1. Define your folders
    input_folder = "source_videos"          # Where your long 4-min videos are
    output_folder = "data/1_Raw_Videos"     # Where the 15-second clips will go
    clip_length = 15

    # Create the output folder if it doesn't exist yet
    os.makedirs(output_folder, exist_ok=True)

    # Find all .mp4 videos in your input folder
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    
    if not video_files:
        print(f"No .mp4 files found in '{input_folder}'. Check your folder names!")
        return

    print(f"Found {len(video_files)} long video(s). Starting the slicer...\n")

    global_clip_count = 1 # This keeps the numbering going (V01, V02, V03...)

    # Loop through every long video you put in the folder
    for video_path in sorted(video_files):
        print(f"Opening: {os.path.basename(video_path)}")
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # Calculate exactly how many full 15-second clips are in this specific video
        num_clips = int(duration // clip_length)
        
        for i in range(num_clips):
            start_time = i * clip_length
            end_time = start_time + clip_length
            
            # Cut the clip
            subclip = video.subclipped(start_time, end_time)
            
            # Name it sequentially
            filename = f"V{global_clip_count:02d}_DeepSea.mp4"
            output_path = os.path.join(output_folder, filename)
            
            print(f"  --> Saving {filename} (from {start_time}s to {end_time}s)")
            
            # Save it (logger=None stops it from printing huge loading bars)
            subclip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
            
            global_clip_count += 1
            
        video.close()
        print(f"Finished slicing {os.path.basename(video_path)}\n")

    print(f"--- SUCCESS! {global_clip_count - 1} clips saved to '{output_folder}' ---")

if __name__ == "__main__":
    slice_all_videos()