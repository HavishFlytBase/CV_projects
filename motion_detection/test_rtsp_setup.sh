#!/bin/bash

# RTSP Motion Detection Test Setup Script

echo "🚀 Setting up RTSP Motion Detection Test Environment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VIDEO_PATH="/Users/havishbychapur/Movies/chandeRoadTwoWheelersFourWheelers.mp4"
INPUT_RTSP="rtsp://localhost:8554/input"
METHOD="frame_differencing"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Cleaning up...${NC}"
    
    # # Stop MediaMTX container
    # docker stop mediamtx-motion-test 2>/dev/null || true
    
    # Kill any remaining processes
    pkill -f "ffmpeg.*rtsp://localhost:8554" 2>/dev/null || true
    pkill -f "rtsp_motion_detection.py" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Check dependencies
echo -e "${BLUE}🔍 Checking dependencies...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}❌ FFmpeg not found. Please install FFmpeg first.${NC}"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo -e "${RED}❌ Video file not found: $VIDEO_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies found${NC}"

# Check if MediaMTX is accessible
echo -e "\n${BLUE}� Checking MediaMTX server...${NC}"
if nc -z localhost 8554 2>/dev/null; then
    echo -e "${GREEN}✅ MediaMTX is accessible on port 8554${NC}"
else
    echo -e "${RED}❌ MediaMTX is not accessible on port 8554${NC}"
    echo -e "${YELLOW}Please ensure MediaMTX is running: docker run --rm -d -p 8554:8554 -p 9997:9997 bluenviron/mediamtx${NC}"
    exit 1
fi

# Step 3: Start input video stream with better RTSP publishing
echo -e "\n${BLUE}📹 Starting input video stream...${NC}"
ffmpeg -re -stream_loop -1 -i "$VIDEO_PATH" \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -g 30 -keyint_min 30 -sc_threshold 0 \
    -f rtsp -rtsp_transport tcp "$INPUT_RTSP" > /tmp/ffmpeg_input.log 2>&1 &

FFMPEG_PID=$!
sleep 8

# Check if FFmpeg is still running
if ps -p $FFMPEG_PID > /dev/null; then
    echo -e "${GREEN}✅ Input stream started successfully${NC}"
    
    # Test if stream is accessible
    timeout 5 ffprobe "$INPUT_RTSP" -v quiet -select_streams v:0 -show_entries stream=width,height 2>/dev/null && \
        echo -e "${GREEN}✅ Stream is accessible and ready${NC}" || \
        echo -e "${YELLOW}⚠️  Stream may still be initializing${NC}"
else
    echo -e "${RED}❌ Failed to start input stream${NC}"
    echo -e "${YELLOW}FFmpeg logs:${NC}"
    cat /tmp/ffmpeg_input.log | tail -10
    exit 1
fi

# Step 4: Start motion detection
echo -e "\n${BLUE}🎯 Starting motion detection...${NC}"
python3 rtsp_motion_detection.py \
    --input "$INPUT_RTSP" \
    --method "$METHOD" > /tmp/motion_detection.log 2>&1 &

MOTION_PID=$!
sleep 5

if ps -p $MOTION_PID > /dev/null; then
    echo -e "${GREEN}✅ Motion detection started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start motion detection${NC}"
    echo -e "${YELLOW}Motion detection logs:${NC}"
    cat /tmp/motion_detection.log | tail -10
    exit 1
fi

# Show URLs and instructions
echo -e "\n${YELLOW}============================================${NC}"
echo -e "${YELLOW}🔗 RTSP STREAM & MOTION DETECTION READY:${NC}"
echo -e "${YELLOW}============================================${NC}"
echo -e "${GREEN}📥 Input Stream:  ${INPUT_RTSP}${NC}"
echo -e "${GREEN}🎯 Motion Detection: Running in OpenCV window${NC}"
echo -e "\n${BLUE}💡 To view input stream:${NC}"
echo -e "   VLC: Media -> Open Network Stream"
echo -e "   FFplay: ffplay ${INPUT_RTSP}"
echo -e "\n${BLUE}📊 MediaMTX API: http://localhost:9997${NC}"
echo -e "${YELLOW}============================================${NC}"

echo -e "\n${GREEN}🚀 All systems running! Press Ctrl+C to stop.${NC}"

# Monitor processes
while true; do
    sleep 10
    
    # Check if processes are still running
    if ! ps -p $FFMPEG_PID > /dev/null; then
        echo -e "${RED}⚠️  Input stream stopped${NC}"
    fi
    
    if ! ps -p $MOTION_PID > /dev/null; then
        echo -e "${RED}⚠️  Motion detection stopped${NC}"
    fi
    
    # Check if MediaMTX port is still accessible
    if ! nc -z localhost 8554 2>/dev/null; then
        echo -e "${RED}⚠️  MediaMTX server not accessible${NC}"
    fi
done
