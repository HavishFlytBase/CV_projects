#!/usr/bin/env python3
"""
Test script for GPU-accelerated motion detection
"""

import cv2
import numpy as np
import time
import os
from mp4_motion_detection_gpu import detect_motion_from_mp4, gpu_manager

def create_test_video(output_path, width=640, height=480, fps=30, duration=3):
    """Create a test video with moving objects"""
    print(f"Creating test video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # Dark gray background
        
        # Add noise
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Moving white rectangle
        x_pos = int((frame_num / total_frames) * (width - 100))
        y_pos = height // 3
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 100, y_pos + 60), (255, 255, 255), -1)
        
        # Moving red circle
        circle_x = width - int((frame_num / total_frames) * (width - 50))
        circle_y = 2 * height // 3
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 0, 255), -1)
        
        # Moving blue triangle
        triangle_x = int(width // 2 + 100 * np.sin(frame_num * 0.2))
        triangle_y = int(height // 2 + 50 * np.cos(frame_num * 0.1))
        pts = np.array([[triangle_x, triangle_y - 25], 
                       [triangle_x - 25, triangle_y + 25], 
                       [triangle_x + 25, triangle_y + 25]], np.int32)
        cv2.fillPoly(frame, [pts], (255, 0, 0))
        
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created: {total_frames} frames")
    return output_path

def benchmark_methods(video_path, methods_to_test):
    """Benchmark GPU vs CPU performance"""
    print("\n🏁 PERFORMANCE BENCHMARK")
    print("="*70)
    
    results = {}
    
    for method in methods_to_test:
        print(f"\n🧪 Testing {method}...")
        
        results[method] = {}
        
        # Test GPU version
        if gpu_manager.is_gpu_enabled():
            print("  📊 GPU Processing...")
            try:
                start_time = time.time()
                gpu_detections = detect_motion_from_mp4(video_path, method=method, use_gpu=True, output_path=None)
                gpu_time = time.time() - start_time
                
                if gpu_detections is not None:
                    gpu_total_detections = sum(len(d) for d in gpu_detections)
                    gpu_frames_with_detections = sum(1 for d in gpu_detections if len(d) > 0)
                    
                    results[method]['gpu'] = {
                        'time': gpu_time,
                        'total_detections': gpu_total_detections,
                        'frames_with_detections': gpu_frames_with_detections,
                        'fps': len(gpu_detections) / gpu_time if gpu_time > 0 else 0
                    }
                    print(f"    ✅ GPU: {gpu_frames_with_detections}/{len(gpu_detections)} frames, {gpu_total_detections} detections, {gpu_time:.2f}s, {results[method]['gpu']['fps']:.1f} FPS")
                else:
                    print("    ❌ GPU: Failed")
                    results[method]['gpu'] = None
                    
            except Exception as e:
                print(f"    ❌ GPU: Error - {e}")
                results[method]['gpu'] = None
        else:
            print("    ⚠️  GPU: Not available")
            results[method]['gpu'] = None
        
        # Test CPU version
        print("  📊 CPU Processing...")
        try:
            start_time = time.time()
            cpu_detections = detect_motion_from_mp4(video_path, method=method, use_gpu=False, output_path=None)
            cpu_time = time.time() - start_time
            
            if cpu_detections is not None:
                cpu_total_detections = sum(len(d) for d in cpu_detections)
                cpu_frames_with_detections = sum(1 for d in cpu_detections if len(d) > 0)
                
                results[method]['cpu'] = {
                    'time': cpu_time,
                    'total_detections': cpu_total_detections,
                    'frames_with_detections': cpu_frames_with_detections,
                    'fps': len(cpu_detections) / cpu_time if cpu_time > 0 else 0
                }
                print(f"    ✅ CPU: {cpu_frames_with_detections}/{len(cpu_detections)} frames, {cpu_total_detections} detections, {cpu_time:.2f}s, {results[method]['cpu']['fps']:.1f} FPS")
            else:
                print("    ❌ CPU: Failed")
                results[method]['cpu'] = None
                
        except Exception as e:
            print(f"    ❌ CPU: Error - {e}")
            results[method]['cpu'] = None
        
        # Calculate speedup
        if (results[method].get('gpu') and results[method].get('cpu') and 
            results[method]['gpu']['time'] > 0 and results[method]['cpu']['time'] > 0):
            speedup = results[method]['cpu']['time'] / results[method]['gpu']['time']
            print(f"    🚀 Speedup: {speedup:.2f}x")
            results[method]['speedup'] = speedup
        else:
            results[method]['speedup'] = None
    
    return results

def print_final_report(results):
    """Print final benchmark report"""
    print("\n" + "="*70)
    print("🏆 FINAL BENCHMARK REPORT")
    print("="*70)
    
    print(f"{'Method':<20} {'CPU Time':<10} {'GPU Time':<10} {'Speedup':<10} {'Status':<15}")
    print("-" * 70)
    
    total_speedup = []
    
    for method, result in results.items():
        cpu_time = result.get('cpu', {}).get('time', 0) if result.get('cpu') else 0
        gpu_time = result.get('gpu', {}).get('time', 0) if result.get('gpu') else 0
        speedup = result.get('speedup', 0) or 0
        
        cpu_str = f"{cpu_time:.2f}s" if cpu_time > 0 else "Failed"
        gpu_str = f"{gpu_time:.2f}s" if gpu_time > 0 else "Failed"
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        
        if speedup > 0:
            status = "✅ GPU Faster" if speedup > 1 else "⚠️  CPU Faster"
            total_speedup.append(speedup)
        else:
            status = "❌ Error"
        
        print(f"{method:<20} {cpu_str:<10} {gpu_str:<10} {speedup_str:<10} {status:<15}")
    
    print("-" * 70)
    
    if total_speedup:
        avg_speedup = np.mean(total_speedup)
        print(f"Average GPU Speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 2.0:
            print("🎉 Excellent GPU acceleration!")
        elif avg_speedup > 1.2:
            print("👍 Good GPU acceleration")
        else:
            print("⚠️  Limited GPU benefit - consider CPU processing")
    else:
        print("❌ No successful GPU accelerations")

def main():
    """Main test function"""
    print("🎯 GPU MOTION DETECTION PERFORMANCE TEST")
    print("="*70)
    
    # Check GPU status
    print("🔧 System Status:")
    print(f"- GPU Available: {gpu_manager.gpu_available}")
    print(f"- CUDA Devices: {gpu_manager.cuda_device_count}")
    print(f"- GPU Enabled: {gpu_manager.is_gpu_enabled()}")
    
    # Create test video
    test_video_path = '/tmp/gpu_motion_test.mp4'
    create_test_video(test_video_path)
    
    try:
        # Methods to test
        methods_to_test = ['background_subtraction', 'frame_differencing', 'optical_flow']
        
        # Run benchmark
        results = benchmark_methods(test_video_path, methods_to_test)
        
        # Print final report
        print_final_report(results)
        
        # Test GPU vs CPU with output video
        print("\n" + "="*70)
        print("🎬 CREATING SAMPLE OUTPUT VIDEOS")
        print("="*70)
        
        sample_method = 'optical_flow'
        if gpu_manager.is_gpu_enabled():
            print(f"\n📹 Creating GPU output video ({sample_method})...")
            gpu_output_path = '/tmp/gpu_motion_output.mp4'
            gpu_detections = detect_motion_from_mp4(
                test_video_path, 
                method=sample_method, 
                use_gpu=True, 
                output_path=gpu_output_path
            )
            if os.path.exists(gpu_output_path):
                print(f"✅ GPU output saved: {gpu_output_path}")
        
        print(f"\n📹 Creating CPU output video ({sample_method})...")
        cpu_output_path = '/tmp/cpu_motion_output.mp4'
        cpu_detections = detect_motion_from_mp4(
            test_video_path, 
            method=sample_method, 
            use_gpu=False, 
            output_path=cpu_output_path
        )
        if os.path.exists(cpu_output_path):
            print(f"✅ CPU output saved: {cpu_output_path}")
        
        print("\n" + "="*70)
        print("🔗 INSTALLATION GUIDE FOR FULL GPU SUPPORT")
        print("="*70)
        print("To maximize GPU performance, install these packages:")
        print()
        print("1. OpenCV with CUDA support:")
        print("   pip uninstall opencv-python opencv-contrib-python")
        print("   pip install opencv-contrib-python")
        print()
        print("2. CuPy for GPU array operations:")
        print("   pip install cupy-cuda11x  # or cupy-cuda12x")
        print()
        print("3. GPU-accelerated libraries:")
        print("   pip install numba[cuda]")
        print("   pip install cuml rapids-cudf")
        print()
        print("4. Verify GPU support:")
        print("   python -c \"import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())\"")
        
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
            print(f"\n🧹 Cleaned up test video")

if __name__ == "__main__":
    main()
