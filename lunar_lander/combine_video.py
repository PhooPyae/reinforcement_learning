import os
import subprocess

def combine_videos(directory, output_filename):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    files.sort()  # Make sure files are in the correct order

    # Write file names to a temporary text file
    with open('filelist.txt', 'w') as file:
        for filename in files:
            file.write(f"file '{directory}/{filename}'\n")
    
    # Run ffmpeg to concatenate videos
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'filelist.txt',
        '-c', 'copy',
        output_filename
    ]
    subprocess.run(command)

    # Clean up the temporary file list
    os.remove('filelist.txt')

# Example usage
# combine_videos('lunarlander-agent-random', 'combined_video.mp4')
