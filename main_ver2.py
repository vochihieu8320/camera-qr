import cv2
import easyocr
import re
import time
import numpy as np
import sqlite3
from datetime import datetime
import queue
from tkinter import Tk, Label, Button, Frame, Canvas, PhotoImage, StringVar, Listbox, Scrollbar, END, Text, BOTH, LEFT, RIGHT, Y
from PIL import Image, ImageTk
import os
from pymodbus.client.sync import ModbusTcpClient

class AutoParkingSystemGUI:
    def __init__(self, root, total_spaces=10, plc_ip="192.168.0.1", plc_port=502):
        self.root = root
        self.root.title("Auto Parking System - Vietnam")
        self.root.geometry("1400x900")  # Increased window size
        self.total_spaces = total_spaces
        self.occupied_spaces = set()
        self.last_detected_plate = ""
        self.last_detection_time = 0
        self.frame_skip = 3  # Reduced frame skip for better detection
        self.frame_count = 0
        self.running = True
        self.current_frame = None
        self.detected_plate_image = None
        self.plc_client = self.connect_to_plc(plc_ip, plc_port)

        # Initialize EasyOCR with Vietnamese support
        self.reader = easyocr.Reader(['en'], gpu=True)  # Set to False if no GPU available

        # Initialize SQLite database
        self.init_database()

        # GUI Elements
        self.setup_gui()

        # Start Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error opening webcam")
            exit()
        
        # Set camera resolution for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Load existing parking data
        self.load_existing_parking_data()
        self.update_camera_feed()

    def init_database(self):
        """Initialize SQLite database for parking records"""
        self.conn = sqlite3.connect('parking_system.db', check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT NOT NULL,
                space_number INTEGER,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                status TEXT DEFAULT 'parked'
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_spaces (
                space_number INTEGER PRIMARY KEY,
                is_occupied BOOLEAN DEFAULT 0,
                license_plate TEXT,
                entry_time TIMESTAMP
            )
        ''')

        # Check if image_path column exists, if not add it
        self.cursor.execute("PRAGMA table_info(parking_records)")
        columns = [column[1] for column in self.cursor.fetchall()]
        if 'image_path' not in columns:
            self.cursor.execute('ALTER TABLE parking_records ADD COLUMN image_path TEXT')

        # Initialize parking spaces
        for i in range(1, self.total_spaces + 1):
            self.cursor.execute('''
                INSERT OR IGNORE INTO parking_spaces (space_number, is_occupied) 
                VALUES (?, 0)
            ''', (i,))

        self.conn.commit()

    def setup_gui(self):
        """Set up the GUI layout with larger camera feed"""
        # Main container
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Left side - Camera and detected plate (increased size)
        left_frame = Frame(main_frame, width=900)  # Increased width
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        # Camera Feed
        camera_frame = Frame(left_frame)
        camera_frame.pack(pady=(0, 10))

        camera_title = Label(camera_frame, text="Live Camera Feed", font=("Arial", 18, "bold"))
        camera_title.pack()

        # Much larger camera feed
        self.camera_label = Label(camera_frame, bg="black", width=300, height=300)
        self.camera_label.pack()

        # Detected License Plate Section
        detected_frame = Frame(left_frame)
        detected_frame.pack(fill=BOTH, expand=True)

        # Detected plate info
        plate_info_frame = Frame(detected_frame)
        plate_info_frame.pack(pady=(0, 10))

        self.plate_var = StringVar()
        self.plate_var.set("No plate detected")

        plate_title = Label(plate_info_frame, text="Detected License Plate", font=("Arial", 16, "bold"))
        plate_title.pack()

        self.plate_display = Label(plate_info_frame, textvariable=self.plate_var,
                                    font=("Arial", 24, "bold"), fg="green", bg="white",
                                    relief="sunken", padx=20, pady=10)
        self.plate_display.pack()

        # Detected plate image
        plate_image_frame = Frame(detected_frame)
        plate_image_frame.pack()

        plate_image_title = Label(plate_image_frame, text="Detected Plate Image", font=("Arial", 14, "bold"))
        plate_image_title.pack()

        self.plate_image_label = Label(plate_image_frame, bg="gray", width=50, height=20)
        self.plate_image_label.pack()

        # Right side - Parking status and controls
        right_frame = Frame(main_frame, width=400)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # Parking Statistics
        stats_frame = Frame(right_frame)
        stats_frame.pack(pady=(0, 10))

        stats_title = Label(stats_frame, text="Parking Statistics", font=("Arial", 16, "bold"))
        stats_title.pack()

        self.stats_frame_content = Frame(stats_frame)
        self.stats_frame_content.pack()

        # Current Parking Status
        status_frame = Frame(right_frame)
        status_frame.pack(fill=BOTH, expand=True)

        self.status_label = Label(status_frame, text="Current Parked Vehicles", font=("Arial", 16, "bold"))
        self.status_label.pack()

        # Scrollable listbox for parked vehicles
        listbox_frame = Frame(status_frame)
        listbox_frame.pack(fill=BOTH, expand=True)

        self.status_listbox = Listbox(listbox_frame, height=15, width=40, font=("Arial", 12))
        scrollbar = Scrollbar(listbox_frame, orient="vertical")

        self.status_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_listbox.yview)

        self.status_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Control buttons
        control_frame = Frame(right_frame)
        control_frame.pack(pady=10)

        self.refresh_btn = Button(control_frame, text="Refresh Status",
                                command=self.update_parking_status,
                                font=("Arial", 12), bg="lightblue")
        self.refresh_btn.pack(side=LEFT, padx=5)

        self.clear_btn = Button(control_frame, text="Clear All Parking",
                                command=self.clear_all_parking,
                                font=("Arial", 12), bg="lightcoral")
        self.clear_btn.pack(side=LEFT, padx=5)

        # Initialize displays
        self.update_parking_status()
        self.update_parking_stats()

    def load_existing_parking_data(self):
        """Load existing parking data from database"""
        self.cursor.execute('SELECT space_number FROM parking_spaces WHERE is_occupied = 1')
        occupied = self.cursor.fetchall()
        for space, in occupied:
            self.occupied_spaces.add(space)

    def update_parking_status(self):
        """Update the parking status listbox"""
        self.status_listbox.delete(0, END)
        self.cursor.execute('''
            SELECT space_number, license_plate, entry_time 
            FROM parking_spaces 
            WHERE is_occupied = 1 
            ORDER BY space_number
        ''')
        occupied = self.cursor.fetchall()
        
        for space, plate, entry_time in occupied:
            # Format entry time
            if entry_time:
                entry_dt = datetime.fromisoformat(entry_time)
                time_str = entry_dt.strftime("%H:%M:%S")
            else:
                time_str = "Unknown"
            
            self.status_listbox.insert(END, f"Space {space:2d}: {plate} (Entry: {time_str})")
        
        if not occupied:
            self.status_listbox.insert(END, "No vehicles currently parked")

    def update_parking_stats(self):
        """Update parking statistics display"""
        # Clear existing stats
        for widget in self.stats_frame_content.winfo_children():
            widget.destroy()

        # Get current stats
        occupied_count = len(self.occupied_spaces)
        available_count = self.total_spaces - occupied_count
        occupancy_rate = (occupied_count / self.total_spaces) * 100 if self.total_spaces > 0 else 0

        # Display stats
        stats_text = [
            f"Total Spaces: {self.total_spaces}",
            f"Occupied: {occupied_count}",
            f"Available: {available_count}",
            f"Occupancy Rate: {occupancy_rate:.1f}%"
        ]

        for i, stat in enumerate(stats_text):
            color = "red" if "Occupied" in stat and occupied_count > 0 else "green" if "Available" in stat else "blue"
            stat_label = Label(self.stats_frame_content, text=stat, font=("Arial", 12, "bold"), fg=color)
            stat_label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

    def preprocess_image_for_vietnamese_plates(self, image):
        """Enhanced preprocessing specifically for Vietnamese license plates"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 3)
        
        # If the image is mostly dark, invert it
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
        
        return thresh

    def correct_vietnamese_ocr_errors(self, text):
        """Fix common OCR errors for Vietnamese license plates"""
        # Vietnamese license plates use specific characters
        corrections = {
            # Numbers that look like letters
            '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G',
            # Letters that look like numbers
            'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6',
            # Common OCR mistakes
            'Q': '0', 'Z': '2', 'l': '1', 'o': '0',
        }
        
        # Remove any non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Apply corrections
        corrected = ''.join(corrections.get(c, c) for c in cleaned)
        
        return corrected

    def validate_vietnamese_license_plate(self, plate):
        """Validate Vietnamese license plate format"""
        if not plate or len(plate) < 8 or len(plate) > 9:
            return False
        
        # Vietnamese license plate patterns:
        # Format 1: 12A-34567 (8 characters without dash)
        # Format 2: 12A-345678 (9 characters without dash)
        # Format 3: 12AB-34567 (9 characters without dash)
        
        # Remove any dashes or spaces
        clean_plate = re.sub(r'[-\s]', '', plate)
        
        # Check if it's 8 or 9 characters
        if len(clean_plate) not in [8, 9]:
            return False
        
        # Check patterns
        patterns = [
            r'^[0-9]{2}[A-Z]{1}[0-9]{5}$',  # 12A34567 (8 chars)
            r'^[0-9]{2}[A-Z]{1}[0-9]{6}$',  # 12A345678 (9 chars)
            r'^[0-9]{2}[A-Z]{2}[0-9]{5}$',  # 12AB34567 (9 chars)
        ]
        
        return any(re.match(pattern, clean_plate) for pattern in patterns)

    def detect_license_plate(self, frame):
        """Enhanced license plate detection for Vietnamese plates"""
        # Multiple preprocessing approaches
        processed_frames = []
        
        # Original preprocessing
        processed_frames.append(self.preprocess_image_for_vietnamese_plates(frame))
        
        # Alternative preprocessing with different parameters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frames.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2))
        
        # Edge detection approach
        edges = cv2.Canny(gray, 50, 150)
        processed_frames.append(edges)
        
        best_plate = None
        best_confidence = 0
        best_plate_region = None
        
        # Try detection on each processed frame
        for processed_frame in processed_frames:
            try:
                # Use EasyOCR with specific settings for license plates
                detected_plates = self.reader.readtext(
                    processed_frame,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                    width_ths=0.7,
                    height_ths=0.7,
                    paragraph=False
                )
                
                for detection in detected_plates:
                    bbox, text, confidence = detection
                    
                    # Clean the detected text
                    corrected_text = self.correct_vietnamese_ocr_errors(text)
                    
                    # Validate the plate
                    if (confidence > 0.4 and 
                        self.validate_vietnamese_license_plate(corrected_text) and
                        confidence > best_confidence):
                        
                        best_confidence = confidence
                        best_plate = corrected_text
                        
                        # Extract the region of interest with more padding
                        points = np.array(bbox, dtype=np.int32)
                        x, y, w, h = cv2.boundingRect(points)
                        
                        # Add padding
                        padding = 20
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame.shape[1] - x, w + 2 * padding)
                        h = min(frame.shape[0] - y, h + 2 * padding)
                        
                        best_plate_region = frame[y:y+h, x:x+w]
                        
            except Exception as e:
                print(f"Error in plate detection: {e}")
                continue
        
        return best_plate, best_plate_region

    def update_camera_feed(self):
        """Update the camera feed in the GUI with larger display"""
        if not self.running:
            return

        success, frame = self.cap.read()
        if success:
            self.current_frame = frame.copy()

            # Resize frame for larger display
            resized_frame = cv2.resize(frame, (1000, 750))  # Much larger resolution
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

            # Process every few frames for plate detection
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                # Use original full resolution frame for detection
                plate_text, plate_image = self.detect_license_plate(self.current_frame)
                current_time = time.time()

                if plate_text and plate_text != self.last_detected_plate:
                    # Check if enough time has passed since last detection
                    if current_time - self.last_detection_time > 2:  # 2 second cooldown
                        self.last_detected_plate = plate_text
                        self.last_detection_time = current_time
                        self.plate_var.set(f"Plate: {plate_text}")

                        # Display the detected plate image
                        if plate_image is not None:
                            self.display_plate_image(plate_image)

                        self.write_to_plc(plate_text)

                        # Check if this plate is already parked
                        if not self.is_plate_already_parked(plate_text):
                            # Save the full frame image, not just the plate region
                            self.store_parking_data(plate_text, self.current_frame, plate_image)
                        else:
                            # Vehicle is leaving
                            self.handle_vehicle_exit(plate_text)

        self.root.after(8, self.update_camera_feed)  # Faster update for smoother video

    def display_plate_image(self, plate_image):
        """Display the detected license plate image"""
        if plate_image is not None:
            # Resize the plate image for display
            height, width = plate_image.shape[:2]
            if width > 300:
                scale = 300 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                plate_image = cv2.resize(plate_image, (new_width, new_height))
            
            # Convert to RGB for display
            rgb_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_plate)
            imgtk = ImageTk.PhotoImage(image=img)
            self.plate_image_label.imgtk = imgtk
            self.plate_image_label.configure(image=imgtk)

    def is_plate_already_parked(self, plate):
        """Check if a license plate is already parked"""
        self.cursor.execute('SELECT COUNT(*) FROM parking_spaces WHERE license_plate = ? AND is_occupied = 1', (plate,))
        count = self.cursor.fetchone()[0]
        return count > 0

    def handle_vehicle_exit(self, plate):
        """Handle vehicle exit"""
        self.cursor.execute('''
            UPDATE parking_spaces 
            SET is_occupied = 0, license_plate = NULL, entry_time = NULL 
            WHERE license_plate = ?
        ''', (plate,))
        
        # Update parking records
        self.cursor.execute('''
            UPDATE parking_records 
            SET exit_time = ?, status = 'exited' 
            WHERE license_plate = ? AND status = 'parked'
        ''', (datetime.now(), plate))
        
        self.conn.commit()
        
        # Update occupied spaces
        self.cursor.execute('SELECT space_number FROM parking_spaces WHERE license_plate = ?', (plate,))
        result = self.cursor.fetchone()
        if result:
            space = result[0]
            self.occupied_spaces.discard(space)
        
        self.update_parking_status()
        self.update_parking_stats()
        print(f"Vehicle {plate} has exited the parking lot")

    def store_parking_data(self, plate, frame, plate_image):
        """Store parking data in the database and update the GUI"""
        try:
            # Find next available space
            self.cursor.execute('SELECT space_number FROM parking_spaces WHERE is_occupied = 0 ORDER BY space_number LIMIT 1')
            result = self.cursor.fetchone()
            
            if result:
                space = result[0]
                entry_time = datetime.now()

                # Create images directory if it doesn't exist
                if not os.path.exists('parking_images'):
                    os.makedirs('parking_images')

                # Save FULL frame image (not just plate region)
                image_filename = f"parking_images/full_frame_{plate}_{entry_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(image_filename, frame)

                # Save plate region image if available
                if plate_image is not None:
                    plate_filename = f"parking_images/plate_region_{plate}_{entry_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(plate_filename, plate_image)

                # Update parking_spaces table
                self.cursor.execute('''
                    UPDATE parking_spaces 
                    SET is_occupied = 1, license_plate = ?, entry_time = ?
                    WHERE space_number = ?
                ''', (plate, entry_time, space))

                # Insert into parking_records table
                self.cursor.execute("PRAGMA table_info(parking_records)")
                columns = [column[1] for column in self.cursor.fetchall()]
                
                if 'image_path' in columns:
                    self.cursor.execute('''
                        INSERT INTO parking_records (license_plate, space_number, entry_time, status, image_path)
                        VALUES (?, ?, ?, 'parked', ?)
                    ''', (plate, space, entry_time, image_filename))
                else:
                    self.cursor.execute('''
                        INSERT INTO parking_records (license_plate, space_number, entry_time, status)
                        VALUES (?, ?, ?, 'parked')
                    ''', (plate, space, entry_time))

                self.conn.commit()

                # Update occupied spaces
                self.occupied_spaces.add(space)

                # Update GUI
                self.update_parking_status()
                self.update_parking_stats()
                
                print(f"Vehicle {plate} assigned to space {space}")
            else:
                print("No available parking spaces")
                self.plate_var.set(f"FULL - {plate}")
        
        except Exception as e:
            print(f"Error storing parking data: {e}")
            # Try to rollback any partial changes
            try:
                self.conn.rollback()
            except:
                pass

    def clear_all_parking(self):
        """Clear all parking data (for testing purposes)"""
        self.cursor.execute('UPDATE parking_spaces SET is_occupied = 0, license_plate = NULL, entry_time = NULL')
        self.cursor.execute('UPDATE parking_records SET exit_time = ?, status = "cleared" WHERE status = "parked"', (datetime.now(),))
        self.conn.commit()
        
        self.occupied_spaces.clear()
        self.update_parking_status()
        self.update_parking_stats()
        print("All parking spaces cleared")

    def shutdown(self):
        """Shutdown the system"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.conn:
            self.conn.close()
        self.root.destroy()

    def connect_to_plc(self, ip, port):
        """Connect to the PLC using Modbus TCP"""
        try:
            client = ModbusTcpClient(ip, port)
            if client.connect():
                print(f"Connected to PLC at {ip}:{port}")
                return client
            else:
                print(f"Failed to connect to PLC at {ip}:{port}")
                return None
        except Exception as e:
            print(f"Error connecting to PLC: {e}")
            return None

    def write_to_plc(self, license_plate):
        """Write the license plate to the PLC"""
        if self.plc_client:
            try:
                # Convert license plate to ASCII values and pad to 16 characters
                ascii_values = [ord(c) for c in license_plate]
                while len(ascii_values) < 16:
                    ascii_values.append(0)  # Pad with zeros

                # Write to PLC (e.g., starting at register 0)
                self.plc_client.write_registers(0, ascii_values)
                print(f"License plate '{license_plate}' written to PLC")
            except Exception as e:
                print(f"Error writing to PLC: {e}")
        else:
            print("PLC client is not connected")

def main():
    root = Tk()
    app = AutoParkingSystemGUI(root, total_spaces=10)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    root.mainloop()

if __name__ == "__main__":
    main()
