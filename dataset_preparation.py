import os
import json
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm


def video_to_landmarks(video_path: str, output_path: str, 
                       num_frames: int = 30, target_size=(224, 224)) -> bool:
    """
    Convert a video file to landmarks sequence.
    
    Args:
        video_path: Path to input video
        output_path: Path to save .npy landmarks file
        num_frames: Number of frames to extract
        target_size: Frame resize dimensions
    
    Returns:
        True if successful, False otherwise
    """
    cap = cv2.VideoCapture(video_path)
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False
    )
    
    landmarks_sequence = []
    
    while len(landmarks_sequence) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, target_size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(frame_rgb)
        
        landmarks = []
        
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        landmarks_sequence.append(landmarks)
    
    cap.release()
    holistic.close()
    
    if len(landmarks_sequence) == 0:
        return False
    
    sequence = np.array(landmarks_sequence)
    
    if len(sequence) < num_frames:
        padding = np.zeros((num_frames - len(sequence), 126))
        sequence = np.vstack([sequence, padding])
    else:
        sequence = sequence[:num_frames]
    
    np.save(output_path, sequence)
    return True


def dataset_folder_to_npy(input_dir: str, output_dir: str, 
                          num_frames: int = 30, extensions=('.mp4', '.avi', '.mov')):
    """
    Convert entire folder structure to .npy files.
    
    Expected input structure:
        input_dir/
            class1/
                video1.mp4
                video2.mp4
            class2/
                video1.mp4
    
    Output structure:
        output_dir/
            class1/
                video1.npy
                video2.npy
            class2/
                video1.npy
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_output = output_path / class_dir.name
        class_output.mkdir(exist_ok=True)
        
        for video_file in class_dir.glob('*'):
            if video_file.suffix.lower() not in extensions:
                continue
            
            output_file = class_output / f"{video_file.stem}.npy"
            
            if output_file.exists():
                print(f"Skipping {video_file.name} (already exists)")
                continue
            
            print(f"Processing: {video_file.name}")
            video_to_landmarks(str(video_file), str(output_file), num_frames)


def webcam_capture(output_dir: str, class_name: str, num_samples: int = 10,
                   seconds_per_sample: int = 3, camera_index: int = 0):
    """
    Capture samples from webcam for a specific class.
    
    Args:
        output_dir: Output directory
        class_name: Name of the sign class
        num_samples: Number of samples to capture
        seconds_per_sample: Duration of each sample in seconds
        camera_index: Camera device index
    """
    output_path = Path(output_dir) / class_name
    output_path.mkdir(exist_ok=True, parents=True)
    
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True
    )
    
    sample_count = 0
    
    print(f"Collecting {num_samples} samples for class '{class_name}'")
    print("Press SPACE to capture a sample, 'Q' to quit early")
    
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        landmarks_sequence = []
        frames_recorded = 0
        fps = 10
        total_frames = seconds_per_sample * fps
        
        print(f"Sample {sample_count + 1}/{num_samples} - Press SPACE to start recording...")
        
        while frames_recorded < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            landmarks = []
            
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0] * 63)
            
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0] * 63)
            
            landmarks_sequence.append(landmarks)
            frames_recorded += 1
            
            cv2.putText(frame, f"Recording: {frames_recorded}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Recording', frame)
            cv2.waitKey(100)
        
        if len(landmarks_sequence) > 0:
            sequence = np.array(landmarks_sequence)
            
            if len(sequence) < 30:
                padding = np.zeros((30 - len(sequence), 126))
                sequence = np.vstack([sequence, padding])
            else:
                sequence = sequence[:30]
            
            output_file = output_path / f"sample_{sample_count:03d}.npy"
            np.save(output_file, sequence)
            sample_count += 1
            print(f"Saved: {output_file}")
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Done!")


def create_class_labels(dataset_dir: str, output_path: str):
    """Create class_labels.json from folder structure."""
    dataset_path = Path(dataset_dir)
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    with open(output_path, 'w') as f:
        json.dump(classes, f, indent=2)
    
    print(f"Created {output_path} with {len(classes)} classes")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset preparation tools")
    parser.add_argument('mode', choices=['convert-folder', 'capture', 'create-labels'],
                       help='Mode to run')
    parser.add_argument('--input', '-i', help='Input directory or video path')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--class-name', help='Class name for webcam capture')
    parser.add_argument('--num-samples', '-n', type=int, default=10,
                       help='Number of samples to capture')
    parser.add_argument('--num-frames', type=int, default=30,
                       help='Number of frames per sample')
    
    args = parser.parse_args()
    
    if args.mode == 'convert-folder':
        dataset_folder_to_npy(args.input, args.output, args.num_frames)
        create_class_labels(args.output, Path(args.output) / 'class_labels.json')
    
    elif args.mode == 'capture':
        if not args.class_name:
            print("Error: --class-name required for capture mode")
        else:
            webcam_capture(args.output, args.class_name, args.num_samples)
    
    elif args.mode == 'create-labels':
        create_class_labels(args.input, args.output)