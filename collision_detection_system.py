import cv2
import numpy as np
import time
import pygame
from ultralytics import YOLO
import math
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Initialize pygame for audio alerts
pygame.init()
pygame.mixer.init()

class KalmanFilterTracker:
    """
    Kalman Filter-based object tracker for vehicle tracking
    """
    def __init__(self, bbox, class_name, id):
        self.id = id
        self.class_name = class_name
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.history = []
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure x, y, width, height)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise covariance
        self.kf.R = np.eye(4) * 10
        
        # Process noise covariance
        q = np.eye(7) * 0.1
        self.kf.Q = block_diag(q, q)
        
        # Initial state covariance
        self.kf.P = np.eye(7) * 100
        
        # Initialize state with bounding box
        x, y, w, h = bbox
        self.kf.x = np.array([x, y, w, h, 0, 0, 0]).reshape(-1, 1)
        
        # Last predicted state
        self.last_prediction = np.array([x, y, w, h])
        
        # Store velocity history for speed calculation
        self.velocity_history = []
        self.positions = [(x, y)]
        self.timestamps = [time.time()]
        self.speed = 0  # Speed in pixels per second
        self.real_world_speed = 0  # Speed in km/h
        
        # Distance estimation
        self.distance = 0  # Distance in meters
        self.time_to_collision = float('inf')  # Time to collision in seconds

    def update(self, bbox):
        """Update the tracker with a new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Update Kalman filter with new measurement
        x, y, w, h = bbox
        self.kf.update(np.array([x, y, w, h]))
        
        # Store position and timestamp for velocity calculation
        self.positions.append((x, y))
        self.timestamps.append(time.time())
        
        # Keep only the last 10 positions for velocity calculation
        if len(self.positions) > 10:
            self.positions.pop(0)
            self.timestamps.pop(0)
        
        # Calculate velocity if we have at least 2 positions
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[0][0]
            dy = self.positions[-1][1] - self.positions[0][1]
            dt = self.timestamps[-1] - self.timestamps[0]
            
            if dt > 0:
                velocity = math.sqrt(dx**2 + dy**2) / dt
                self.velocity_history.append(velocity)
                
                # Keep only the last 5 velocity measurements
                if len(self.velocity_history) > 5:
                    self.velocity_history.pop(0)
                
                # Average velocity
                self.speed = sum(self.velocity_history) / len(self.velocity_history)
                
                # Convert pixel speed to real-world speed (assuming calibration)
                # This is a simplified conversion - in a real system, proper camera calibration would be used
                # Assuming 1 pixel = 0.05 meters at the reference distance
                self.real_world_speed = self.speed * 0.05 * 3.6  # Convert to km/h
        
        return self.kf.x

    def predict(self):
        """Predict the next state"""
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        
        # Extract the predicted bounding box
        x, y, w, h = self.kf.x[0:4, 0]
        self.last_prediction = np.array([x, y, w, h])
        
        return self.last_prediction
    
    def estimate_distance(self, frame_height, focal_length=800, real_height=1.5):
        """
        Estimate distance to object using the pinhole camera model
        
        Parameters:
        - frame_height: Height of the frame in pixels
        - focal_length: Focal length of the camera in pixels (estimated)
        - real_height: Real height of the object in meters (estimated for a car)
        
        Returns:
        - Distance in meters
        """
        _, _, _, h = self.last_prediction
        
        # Avoid division by zero
        if h <= 0:
            return float('inf')
        
        # Calculate distance using pinhole camera model
        # Distance = (real height * focal length) / apparent height
        self.distance = (real_height * focal_length) / h
        
        return self.distance
    
    def estimate_time_to_collision(self, ego_speed=0):
        """
        Estimate time to collision based on distance and relative speed
        
        Parameters:
        - ego_speed: Speed of the ego vehicle in km/h
        
        Returns:
        - Time to collision in seconds
        """
        # Convert ego_speed from km/h to m/s
        ego_speed_ms = ego_speed / 3.6
        
        # Convert object speed from km/h to m/s
        obj_speed_ms = self.real_world_speed / 3.6
        
        # Calculate relative speed (negative means approaching)
        relative_speed = obj_speed_ms - ego_speed_ms
        
        # If objects are moving apart or at the same speed, no collision will occur
        if relative_speed >= 0:
            self.time_to_collision = float('inf')
        else:
            # Time to collision = distance / |relative speed|
            self.time_to_collision = self.distance / abs(relative_speed)
        
        return self.time_to_collision

class RiskAssessmentEnvironment:
    """
    Environment for risk assessment based on object parameters
    """
    def __init__(self, data=None):
        # Define state space
        self.state_size = 5  # distance, rel_velocity, angle, vehicle_speed, weather
        
        # Define action space (risk levels)
        self.action_size = 4  # Low, Medium, High, Critical
        self.risk_levels = ['Low', 'Medium', 'High', 'Critical']
        
        # Current state
        self.state = None
        
        # Risk thresholds
        self.distance_threshold = 20  # meters
        self.ttc_threshold = 3.0  # seconds
        
    def reset(self):
        """Reset the environment"""
        self.state = np.zeros(self.state_size)
        return self.state
        
    def step(self, action):
        """Take an action and return the new state, reward, and done flag"""
        # In a real system, this would update based on the action taken
        # For our purposes, we'll just return the current state
        
        # Calculate reward based on action
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = False
        
        return self.state, reward, done
    
    def _calculate_reward(self, action):
        """Calculate reward based on action and state"""
        # Extract state variables
        distance = self.state[0]
        rel_velocity = self.state[1]
        
        # Calculate time to collision (TTC)
        if rel_velocity < 0:  # Objects approaching each other
            ttc = distance / abs(rel_velocity)
        else:
            ttc = float('inf')
        
        # Determine appropriate risk level based on TTC and distance
        if ttc < 1.0 or distance < 5:
            appropriate_action = 3  # Critical
        elif ttc < 2.0 or distance < 10:
            appropriate_action = 2  # High
        elif ttc < 3.0 or distance < 20:
            appropriate_action = 1  # Medium
        else:
            appropriate_action = 0  # Low
        
        # Reward is higher if the action matches the appropriate action
        if action == appropriate_action:
            reward = 1.0
        else:
            reward = -abs(action - appropriate_action)
        
        return reward
    
    def _get_observation(self):
        """Get the current observation"""
        return self.state

class DQNAgent:
    """
    Deep Q-Network agent for risk assessment
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build a neural network model for DQN"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action based on the current state"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the model with experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)

class AdaptiveWarningSystem:
    """
    System for generating warnings based on risk assessment
    """
    def __init__(self):
        # Load alert sounds
        try:
            self.alert_sounds = {
                'Low': pygame.mixer.Sound('low_alert.wav'),
                'Medium': pygame.mixer.Sound('medium_alert.wav'),
                'High': pygame.mixer.Sound('high_alert.wav'),
                'Critical': pygame.mixer.Sound('critical_alert.wav')
            }
        except:
            # If sound files don't exist, create placeholder sounds of different frequencies
            self.alert_sounds = {}
            print("Warning: Alert sound files not found. Using placeholder sounds.")
    
    def generate_alert(self, risk_level, risk_score, object_info):
        """
        Generate an alert based on risk level and object information
        
        Parameters:
        - risk_level: Risk level (Low, Medium, High, Critical)
        - risk_score: Numerical risk score
        - object_info: Dictionary containing object information
        
        Returns:
        - Alert dictionary
        """
        # Define color based on risk level
        color_mapping = {
            'Low': (0, 255, 0),      # Green
            'Medium': (0, 255, 255),  # Yellow
            'High': (0, 165, 255),   # Orange
            'Critical': (0, 0, 255)  # Red
        }
        
        color = color_mapping.get(risk_level, (255, 255, 255))
        
        # Create message
        message = f"Risk: {risk_level} ({risk_score:.2f})"
        
        if 'distance' in object_info:
            message += f" - {object_info['distance']:.1f}m"
        
        if 'time_to_collision' in object_info and object_info['time_to_collision'] != float('inf'):
            message += f" - TTC: {object_info['time_to_collision']:.1f}s"
        
        if 'class_name' in object_info:
            message += f" - {object_info['class_name']}"
        
        # Create visual alert
        visual_alert = self._create_visual_alert(risk_level, risk_score, message, color)
        
        # Play sound alert for high and critical risks
        audio_alert = None
        if risk_level in ['High', 'Critical']:
            if risk_level in self.alert_sounds:
                audio_alert = self.alert_sounds[risk_level]
                # Play the sound
                try:
                    pygame.mixer.Sound.play(audio_alert)
                except:
                    pass
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'message': message,
            'visual_alert': visual_alert,
            'audio_alert': audio_alert,
            'color': color
        }
    
    def _create_visual_alert(self, risk_level, risk_score, message, color):
        """Create a visual alert image"""
        # Create a blank image for the alert
        alert_img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        
        # Add colored background based on risk level
        cv2.rectangle(alert_img, (0, 0), (400, 100), color, -1)
        
        # Add text
        cv2.putText(alert_img, message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return alert_img
    
    def display_alert(self, alert):
        """Display the alert (for demonstration purposes)"""
        if alert and 'visual_alert' in alert:
            cv2.imshow('Alert', alert['visual_alert'])
            cv2.waitKey(1)

class CollisionDetectionSystem:
    """
    Integrated system for collision detection and risk assessment
    """
    def __init__(self, yolo_model_path=None):
        # Initialize YOLO model
        if yolo_model_path:
            self.detection_model = YOLO(yolo_model_path)
        else:
            # Use a pre-trained YOLO model from Ultralytics
            self.detection_model = YOLO('yolov8n.pt')
        
        # Initialize risk assessment environment
        self.risk_env = RiskAssessmentEnvironment()
        
        # Initialize DQN agent
        self.risk_agent = DQNAgent(state_size=5, action_size=4)
        
        # Initialize warning system
        self.warning_system = AdaptiveWarningSystem()
        
        # Initialize object trackers
        self.trackers = {}
        self.next_id = 0
        
        # Camera parameters (estimated)
        self.focal_length = 800  # pixels
        
        # Vehicle parameters
        self.ego_speed = 0  # km/h
        
        # Frame counter for tracking
        self.frame_count = 0
        
        # Classes of interest for collision detection
        self.classes_of_interest = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']
    
    def process_frame(self, frame, vehicle_speed=0, weather_condition=1.0, road_type=0):
        """
        Process a single frame from the vehicle's camera
        
        Parameters:
        - frame: Camera frame (image)
        - vehicle_speed: Current vehicle speed in km/h
        - weather_condition: Current weather condition (0-1)
        - road_type: Current road type (0-3)
        
        Returns:
        - Processed frame with detections and warnings
        - List of detected objects with risk assessments
        - Highest risk warning
        """
        self.frame_count += 1
        self.ego_speed = vehicle_speed
        
        # Create a copy of the frame for processing
        processed_frame = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Run object detection
        detections = self.detect_objects(frame)
        
        # Update trackers with new detections
        self.update_trackers(detections, height)
        
        # Predict new locations for all trackers
        for tracker_id in list(self.trackers.keys()):
            tracker = self.trackers[tracker_id]
            
            # Predict next position
            prediction = tracker.predict()
            
            # Remove trackers that have been lost for too long
            if tracker.time_since_update > 10:
                del self.trackers[tracker_id]
                continue
            
            # Estimate distance and time to collision
            tracker.estimate_distance(height, self.focal_length)
            tracker.estimate_time_to_collision(self.ego_speed)
        
        # Assess risk for each tracker
        risk_assessments = []
        highest_risk = None
        highest_risk_score = -1
        
        for tracker_id, tracker in self.trackers.items():
            # Extract object information
            distance = tracker.distance
            rel_velocity = -vehicle_speed / 3.6  # Convert to m/s and assume worst case (approaching)
            if tracker.real_world_speed > 0:
                rel_velocity = (tracker.real_world_speed - vehicle_speed) / 3.6
            
            angle = 0  # Assume head-on for simplicity
            size = 1.0  # Normalized size
            
            # Prepare input for risk model
            risk_input = np.array([
                distance,
                rel_velocity,
                angle,
                vehicle_speed / 3.6,  # Convert to m/s
                weather_condition
            ])
            
            # Update environment state
            self.risk_env.state = risk_input
            
            # Get risk assessment from DQN agent
            risk_action = self.risk_agent.act(risk_input)
            risk_level = self.risk_env.risk_levels[risk_action]
            
            # Calculate risk score
            # Lower distance and TTC means higher risk
            ttc = tracker.time_to_collision
            risk_score = 0
            
            if distance <= 5:
                risk_score += 0.5
            elif distance <= 10:
                risk_score += 0.3
            elif distance <= 20:
                risk_score += 0.1
            
            if ttc <= 1.0:
                risk_score += 0.5
            elif ttc <= 2.0:
                risk_score += 0.3
            elif ttc <= 3.0:
                risk_score += 0.1
            
            # Adjust for weather and road type
            risk_score += (1 - weather_condition) * 0.1
            risk_score += road_type * 0.05
            
            # Normalize risk score to 0-1
            risk_score = min(max(risk_score, 0), 1)
            
            # Override risk level based on calculated score
            if risk_score >= 0.7:
                risk_level = 'Critical'
            elif risk_score >= 0.5:
                risk_level = 'High'
            elif risk_score >= 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Create object info dictionary
            object_info = {
                'id': tracker_id,
                'class_name': tracker.class_name,
                'distance': distance,
                'rel_velocity': rel_velocity,
                'time_to_collision': ttc,
                'box': tracker.last_prediction,
                'speed': tracker.real_world_speed
            }
            
            # Create assessment dictionary
            assessment = {
                'tracker': tracker,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_action': risk_action,
                'object_info': object_info
            }
            
            risk_assessments.append(assessment)
            
            # Update highest risk
            if risk_score > highest_risk_score:
                highest_risk_score = risk_score
                highest_risk = assessment
        
        # Generate warning for highest risk
        highest_risk_warning = None
        if highest_risk:
            highest_risk_warning = self.warning_system.generate_alert(
                highest_risk['risk_level'],
                highest_risk['risk_score'],
                highest_risk['object_info']
            )
        
        # Draw results on the frame
        result_frame = self._draw_results(processed_frame, risk_assessments, highest_risk_warning)
        
        return result_frame, risk_assessments, highest_risk_warning
    
    def detect_objects(self, frame):
        """
        Detect objects in the frame using YOLO
        
        Parameters:
        - frame: Input frame
        
        Returns:
        - List of detections
        """
        # Run YOLO detection
        results = self.detection_model(frame)
        
        # Extract detections
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence
                conf = box.conf[0].cpu().numpy()
                
                # Get class
                cls = int(box.cls[0].cpu().numpy())
                cls_name = self.detection_model.names[cls].lower()
                
                # Only keep classes of interest with high confidence
                if cls_name in self.classes_of_interest and conf > 0.5:
                    # Convert to x, y, w, h format for tracker
                    x, y, w, h = x1, y1, x2-x1, y2-y1
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': conf,
                        'class_name': cls_name
                    })
        
        return detections
    
    def update_trackers(self, detections, frame_height):
        """
        Update trackers with new detections
        
        Parameters:
        - detections: List of detections
        - frame_height: Height of the frame
        """
        # If no trackers yet, initialize with all detections
        if not self.trackers:
            for det in detections:
                self.trackers[self.next_id] = KalmanFilterTracker(
                    det['bbox'],
                    det['class_name'],
                    self.next_id
                )
                self.next_id += 1
            return
        
        # Match detections to existing trackers
        matched_indices = []
        
        # For each detection
        for det in detections:
            best_iou = 0.3  # IOU threshold
            best_tracker_id = None
            
            # Find the tracker with the highest IOU
            for tracker_id, tracker in self.trackers.items():
                # Get predicted box from tracker
                pred_box = tracker.last_prediction
                
                # Calculate IOU
                iou = self._calculate_iou(det['bbox'], pred_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_tracker_id = tracker_id
            
            # If a match is found, update the tracker
            if best_tracker_id is not None:
                self.trackers[best_tracker_id].update(det['bbox'])
                matched_indices.append(best_tracker_id)
            else:
                # Create a new tracker
                self.trackers[self.next_id] = KalmanFilterTracker(
                    det['bbox'],
                    det['class_name'],
                    self.next_id
                )
                self.next_id += 1
        
        # Update trackers that weren't matched
        for tracker_id in list(self.trackers.keys()):
            if tracker_id not in matched_indices:
                self.trackers[tracker_id].predict()
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IOU) between two bounding boxes
        
        Parameters:
        - bbox1: First bounding box (x, y, w, h)
        - bbox2: Second bounding box (x, y, w, h)
        
        Returns:
        - IOU score
        """
        # Convert to x1, y1, x2, y2 format
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IOU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def _draw_results(self, frame, risk_assessments, highest_risk_warning):
        """
        Draw detection boxes, risk levels, and warnings on the frame
        
        Parameters:
        - frame: Input frame
        - risk_assessments: List of risk assessments
        - highest_risk_warning: Highest risk warning
        
        Returns:
        - Processed frame with visualizations
        """
        result_frame = frame.copy()
        
        # Color mapping for risk levels
        color_mapping = {
            'Low': (0, 255, 0),      # Green
            'Medium': (0, 255, 255),  # Yellow
            'High': (0, 165, 255),   # Orange
            'Critical': (0, 0, 255)  # Red
        }
        
        # Draw each detection
        for assessment in risk_assessments:
            tracker = assessment['tracker']
            risk_level = assessment['risk_level']
            risk_score = assessment['risk_score']
            object_info = assessment['object_info']
            
            # Get box coordinates
            x, y, w, h = tracker.last_prediction
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            
            # Get color based on risk level
            color = color_mapping.get(risk_level, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with object class, risk level, and distance
            label = f"{tracker.class_name} ({risk_level}, {risk_score:.2f})"
            
            # Add distance and speed information
            if 'distance' in object_info:
                label += f", {object_info['distance']:.1f}m"
            
            if 'speed' in object_info:
                label += f", {object_info['speed']:.1f}km/h"
            
            # Add time to collision if it's finite
            if 'time_to_collision' in object_info and object_info['time_to_collision'] != float('inf'):
                label += f", TTC: {object_info['time_to_collision']:.1f}s"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw trajectory (last 10 positions)
            if len(tracker.positions) >= 2:
                for i in range(len(tracker.positions) - 1):
                    pt1 = (int(tracker.positions[i][0]), int(tracker.positions[i][1]))
                    pt2 = (int(tracker.positions[i+1][0]), int(tracker.positions[i+1][1]))
                    cv2.line(result_frame, pt1, pt2, color, 1)
        
        # Draw highest risk warning at the top of the frame
        if highest_risk_warning:
            # Create warning text
            warning_text = highest_risk_warning['message']
            
            # Draw warning background
            cv2.rectangle(result_frame, (0, 0), (result_frame.shape[1], 30), highest_risk_warning['color'], -1)
            
            # Draw warning text
            cv2.putText(result_frame, warning_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return result_frame

def process_video(input_path, output_path=None, yolo_model_path=None):
    """
    Process a video file for collision detection
    
    Parameters:
    - input_path: Path to input video file
    - output_path: Path to output video file (optional)
    - yolo_model_path: Path to YOLO model (optional)
    """
    # Initialize collision detection system
    system = CollisionDetectionSystem(yolo_model_path)
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Simulate vehicle speed (in a real system, this would come from the vehicle's CAN bus)
        vehicle_speed = 30.0  # km/h
        
        # Simulate weather condition (1.0 = clear, 0.0 = poor visibility)
        weather_condition = 0.9
        
        # Simulate road type (0 = highway, 1 = urban, 2 = rural, 3 = off-road)
        road_type = 1  # urban
        
        # Process frame
        result_frame, assessments, highest_risk = system.process_frame(
            frame,
            vehicle_speed,
            weather_condition,
            road_type
        )
        
        # Create side-by-side visualization
        side_by_side = np.zeros((height, width*2, 3), dtype=np.uint8)
        side_by_side[:, :width] = frame
        side_by_side[:, width:] = result_frame
        
        # Add dividing line
        cv2.line(side_by_side, (width, 0), (width, height), (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(side_by_side, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(side_by_side, "Processed", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Collision Detection System', side_by_side)
        
        # Write frame to output video
        if writer:
            writer.write(side_by_side)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collision Detection System')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--model', type=str, help='Path to YOLO model file')
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.model)
