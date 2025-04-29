import pygame
import random
import time # for wait/pause
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import math
import joblib # save/load models
import os     # file paths

# Initialize Pygame
pygame.init()

# Screen dimensions and setup
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Streetlight Simulation - 4 Modes (0-3)")

# File Paths
MODEL_DIR = "trained_models" # subdirectory for saved models

# Colors
WHITE = (255, 255, 255); BLACK = (0, 0, 0); GRAY = (169, 169, 169)
DARK_GRAY = (64, 64, 64); DARK_GREEN = (34, 139, 34); BLUE = (0, 0, 255); RED = (255, 0, 0)
LAMP_OFF = GRAY; LAMP_HALF = (255, 255, 128); LAMP_75 = (255, 255, 64); LAMP_FULL = (255, 255, 0)
OVERRIDE_COLOR = (255, 165, 0) # for override text

# Font setup
font = pygame.font.SysFont("Arial", 24); small_font = pygame.font.SysFont("Arial", 20)

# Game States (using range for simplicity)
MENU, PATTERN_SELECT, ML_MENU, ML_TRAIN_PATTERN_SELECT, ML_RUN_PATTERN_SELECT, PERFORM_TRAINING, PERFORM_LOADING, SIMULATION = range(8)

# Traffic patterns
MALL_ROAD, NIGHT_ROAD, OFFICE_ROAD = range(3)
PATTERN_NAMES = ["Mall Road", "Night Road", "Office Road"]

# Mode Constants and Names (0-indexed)
MODE_ALWAYS_ON = 0
MODE_MOTION = 1
MODE_ML_ONLY = 2
MODE_ML_MOTION = 3
MODE_NAMES = ["Always On", "Motion Detection", "Machine Learning", "ML + MD"]

# ML State Definitions
STATE_OFF = 0; STATE_HALF = 1; STATE_75_PERCENT = 2; STATE_FULL = 3
STATE_NAMES = ["OFF", "HALF", "75%", "FULL"]

# Simulation constants
SIMULATION_DURATION = 1      # hours
SECONDS_PER_HOUR = 80      # simulation speed factor
FPS = 60
TIME_WINDOW = 0.01         # hours, for flow rate calc / prediction update freq

# TFR Thresholds for ML States (cars/hr)
TFR_THRESHOLD_HALF = 200; TFR_THRESHOLD_75 = 500; TFR_THRESHOLD_FULL = 800

# Power Levels for ML States (Watts)
POWER_OFF = 0; POWER_HALF = 50; POWER_75 = 75; POWER_FULL = 100

# Training Runs
total_training_runs = 30 # for ML model training

# Helper Function for File Paths
def get_model_filepath(pattern_type):
    # generates relative path like 'trained_models/predictor_mall_road.joblib'
    if 0 <= pattern_type < len(PATTERN_NAMES):
        pattern_name_safe = PATTERN_NAMES[pattern_type].replace(' ', '_').lower()
        filename = f"predictor_{pattern_name_safe}.joblib"
        return os.path.join(MODEL_DIR, filename)
    return None

# Classes
class TrafficFlowMonitor:
    # calculates cars/hr based on recent detections
    def __init__(self): self.cars_in_window=0; self.last_check_time=0; self.current_flow_rate=0
    def record_car(self): self.cars_in_window += 1
    def update_flow_rate(self, current_hour):
        if current_hour >= self.last_check_time + TIME_WINDOW:
            if TIME_WINDOW > 0: self.current_flow_rate = max(0, self.cars_in_window / TIME_WINDOW)
            else: self.current_flow_rate = float('inf') # avoid division by zero
            self.cars_in_window = 0; self.last_check_time = current_hour
        return self.current_flow_rate

class TrafficPattern:
    # defines target traffic flow over time for different scenarios
    def __init__(self, pattern_type):
        self.pattern_type = pattern_type
        # randomize timings slightly for each instance
        if self.pattern_type == MALL_ROAD:
            self.mall_rise_duration=0.4; self.mall_fall_duration=0.4; self.mall_start_rise=random.uniform(0.0, 0.2)
            self.mall_peak_time=self.mall_start_rise+self.mall_rise_duration; self.mall_end_fall=self.mall_peak_time+self.mall_fall_duration
        elif self.pattern_type == NIGHT_ROAD:
            self.night_start_fall=0.1; self.night_end_fall=random.uniform(0.4, 0.5); self.night_fall_duration=self.night_end_fall-self.night_start_fall
            self.night_end_rise=0.9; self.night_start_rise=random.uniform(0.5, 0.6); self.night_rise_duration=self.night_end_rise-self.night_start_rise
        elif self.pattern_type == OFFICE_ROAD:
            self.office_peak1_center=random.uniform(0.25, 0.35); self.office_peak2_center=random.uniform(0.65, 0.75)
            self.office_peak1_duration=random.uniform(0.1, 0.3); self.office_peak2_duration=random.uniform(0.1, 0.3)

    def get_target_flow(self, hour):
        # returns target cars/hr for a given hour based on pattern type and randomized timings
        # uses np.random.normal for some variability around the target mean
        mean_flow=0; std_dev=1 # default std_dev
        if self.pattern_type == MALL_ROAD:
            bf,pf=200,800; bs,ps=200,300 # base/peak flow & std_dev
            if hour<self.mall_start_rise: mf,sd=bf,bs
            elif hour<self.mall_peak_time:
                if self.mall_rise_duration>0: p=(hour-self.mall_start_rise)/self.mall_rise_duration; mf=bf+p*(pf-bf); sd=bs+p*(ps-bs)
                else: mf,sd=pf,ps
            elif hour<self.mall_end_fall:
                if self.mall_fall_duration>0: p=(hour-self.mall_peak_time)/self.mall_fall_duration; mf=pf-p*(pf-bf); sd=ps-p*(ps-bs)
                else: mf,sd=bf,bs
            else: mf,sd=bf,bs
        elif self.pattern_type == NIGHT_ROAD:
            hf,lf=800,200; hs,ls=300,200 # high/low flow & std_dev
            if hour<self.night_start_fall: mf,sd=hf,hs
            elif hour<self.night_end_fall:
                if self.night_fall_duration>0: p=(hour-self.night_start_fall)/self.night_fall_duration; mf=hf-p*(hf-lf); sd=hs-p*(hs-ls)
                else: mf,sd=lf,ls
            elif hour<self.night_start_rise: mf,sd=lf,ls
            elif hour<self.night_end_rise:
                if self.night_rise_duration>0: p=(hour-self.night_start_rise)/self.night_rise_duration; mf=lf+p*(hf-lf); sd=ls+p*(hs-ls)
                else: mf,sd=hf,hs
            else: mf,sd=hf,hs
        elif self.pattern_type == OFFICE_ROAD:
            pf,op=800,200; ps,ops=300,200 # peak/off-peak flow & std_dev
            ip1=abs(hour-self.office_peak1_center)<self.office_peak1_duration/2; ip2=abs(hour-self.office_peak2_center)<self.office_peak2_duration/2
            if ip1 or ip2: mf,sd=pf,ps
            else: mf,sd=op,ops
        sd=max(1,sd); # ensure std_dev > 0
        return max(0, np.random.normal(mf,sd)) # ensure flow >= 0

class TrafficDataCollector:
    # collects (hour, state_label) pairs for ML training
    def __init__(self): self.time_data=[]; self.labels=[]
    def collect_sample(self, hour, flow_rate):
        # determine label based on TFR thresholds
        if flow_rate<TFR_THRESHOLD_HALF: label=STATE_OFF
        elif flow_rate<TFR_THRESHOLD_75: label=STATE_HALF
        elif flow_rate<TFR_THRESHOLD_FULL: label=STATE_75_PERCENT
        else: label=STATE_FULL
        self.time_data.append([hour]); self.labels.append(label)
    def combine_data(self, other): # used to merge data from multiple training runs
        self.time_data.extend(other.time_data); self.labels.extend(other.labels)

class StreetlightPredictor:
    # holds the trained ML model (RandomForest) and scaler
    def __init__(self): self.model=None; self.scaler=None; self.is_trained=False
    def train(self, time_data, labels):
        # trains the model and scaler if enough data/classes exist
        if len(time_data)>10 and len(set(labels))>1: # basic check for valid training
            try:
                self.model=RandomForestClassifier(n_estimators=100, random_state=42); self.scaler=StandardScaler()
                sd=self.scaler.fit_transform(time_data); self.model.fit(sd, labels); self.is_trained=True
                print(f"Training successful. Classes learned: {sorted(list(set(labels)))}"); return True
            except ValueError as e: print(f"Training failed: {e}"); self.is_trained=False; return False
        else: print(f"Training skipped: Insufficient data or only one class present."); print(f"Data points: {len(time_data)}, Unique labels: {len(set(labels))}"); self.is_trained=False; return False
    def predict(self, hour):
        # predicts state (0-3) based on hour using the trained model
        if not self.is_trained or self.model is None or self.scaler is None: print("Warning: Predict called before model/scaler were trained/loaded."); return STATE_OFF
        try: s=np.array([[hour]]); ss=self.scaler.transform(s); ps=self.model.predict(ss)[0]; return int(ps)
        except Exception as e: print(f"Prediction failed: {e}"); return STATE_OFF
    def load_model_data(self, model, scaler): # assigns loaded model/scaler
        self.model=model; self.scaler=scaler; self.is_trained=True

class Streetlamp:
    # represents the streetlight object, handles state, power, energy, drawing
    def __init__(self, x, y):
        self.x=x; self.y=y; self.state=STATE_OFF; self.current_power=POWER_OFF
        self.max_power=POWER_FULL; self.detection_range=400; self.total_energy=0
        # map states to power and color properties
        self.state_properties = {STATE_OFF:{'p':POWER_OFF,'c':LAMP_OFF}, STATE_HALF:{'p':POWER_HALF,'c':LAMP_HALF},
                                 STATE_75_PERCENT:{'p':POWER_75,'c':LAMP_75}, STATE_FULL:{'p':POWER_FULL,'c':LAMP_FULL}}
    def set_state(self, ps): # ps = predicted state
        # updates state and current power based on input state
        if ps in self.state_properties: self.state=ps; self.current_power=self.state_properties[ps]['p']
        else: self.state=STATE_OFF; self.current_power=POWER_OFF # default off if invalid state
    def draw(self):
        # draws the lamp post and bulb with appropriate color/glow
        ph=40; # post height
        pygame.draw.rect(screen,GRAY,(self.x,self.y-ph,10,ph)) # post
        lc=self.state_properties.get(self.state,{'c':LAMP_OFF})['c'] # lamp color
        pygame.draw.circle(screen,lc,(self.x+5,self.y-ph-10),15) # bulb
        if self.state!=STATE_OFF: # add glow if not off
            gr=15+self.current_power*0.2; ga=50+self.current_power*0.5; gc=lc # glow radius, alpha, color
            gs=pygame.Surface((gr*2,gr*2),pygame.SRCALPHA); pygame.draw.circle(gs,(*gc,int(ga)),(gr,gr),gr)
            screen.blit(gs,(self.x+5-gr,self.y-ph-10-gr))
    def calculate_energy(self, current_hour, previous_hour):
        # accumulates energy based on current power and time delta
        if self.current_power>0: he=max(0,current_hour-previous_hour); self.total_energy+=self.current_power*he
    def get_energy_display(self): # formats energy for display (Wh or kWh)
        return f"Energy: {self.total_energy:.1f} Wh" if self.total_energy<1000 else f"Energy: {(self.total_energy/1000):.2f} kWh"
    def get_status_display(self): # formats current state/power for display
        sn=STATE_NAMES[self.state]; return f"Lamp State: {sn} ({self.current_power}W)"

class Car:
    # represents a car object
    def __init__(self):
        self.x=WIDTH; self.y=HEIGHT//2+175; self.speed_mps=random.uniform(10,30); ppm=10 # pixels per meter scale
        # calculate speed in pixels per frame based on simulation scaling
        self.sppf=self.speed_mps*ppm*3600/(SECONDS_PER_HOUR*FPS) if SECONDS_PER_HOUR>0 and FPS>0 else 0
        self.length=25; self.height=8
    def move(self): self.x-=self.sppf # move left based on calculated speed
    def draw(self):
        # draws car body and tires
        pygame.draw.rect(screen,WHITE,(self.x,self.y,self.length,self.height),border_radius=2)
        pygame.draw.rect(screen,BLACK,(self.x,self.y,self.length,self.height),1)
        tr=3; ftx=self.x+self.length*0.8; rtx=self.x+self.length*0.2; ty=self.y+self.height
        pygame.draw.circle(screen,DARK_GRAY,(rtx,ty),tr); pygame.draw.circle(screen,DARK_GRAY,(ftx,ty),tr)

# Drawing Functions
def draw_text(text, font_obj, color, x, y, center=False):
    # helper to draw text easily
    ts = font_obj.render(text, True, color); tr = ts.get_rect()
    if center: tr.center = (x, y)
    else: tr.topleft = (x, y)
    screen.blit(ts, tr)

def draw_menu():
    # draws the main mode selection menu
    screen.fill(BLACK); draw_text("Streetlight Simulation", font, WHITE, WIDTH // 2, HEIGHT // 2 - 175, center=True)
    draw_text(f"Press '0' for Mode 0: {MODE_NAMES[0]}", font, WHITE, WIDTH // 2, HEIGHT // 2 - 75, center=True)
    draw_text(f"Press '1' for Mode 1: {MODE_NAMES[1]}", font, WHITE, WIDTH // 2, HEIGHT // 2 - 25, center=True)
    draw_text(f"Press '2' for Mode 2: {MODE_NAMES[2]}", font, WHITE, WIDTH // 2, HEIGHT // 2 + 25, center=True)
    draw_text(f"Press '3' for Mode 3: {MODE_NAMES[3]}", font, WHITE, WIDTH // 2, HEIGHT // 2 + 75, center=True)
    draw_text("Press ESC to exit simulation at any time", small_font, WHITE, WIDTH // 2, HEIGHT // 2 + 150, center=True); pygame.display.flip()

def draw_ml_menu():
    # draws the ML specific menu (Train or Run)
    screen.fill(BLACK); draw_text("ML Mode Options", font, WHITE, WIDTH // 2, HEIGHT // 2 - 100, center=True)
    draw_text("Press '1' to TRAIN Model for a Pattern", font, WHITE, WIDTH // 2, HEIGHT // 2 - 25, center=True)
    draw_text("Press '2' to RUN Simulation with Trained Model", font, WHITE, WIDTH // 2, HEIGHT // 2 + 25, center=True)
    draw_text("(Modes 2 and 3 use these trained models)", small_font, WHITE, WIDTH // 2, HEIGHT // 2 + 65, center=True)
    draw_text("Press ESC to return to Main Menu", small_font, WHITE, WIDTH // 2, HEIGHT // 2 + 115, center=True); pygame.display.flip()

def draw_pattern_select(purpose="select"):
    # draws the traffic pattern selection screen
    screen.fill(BLACK); pt_text = "Train Model For Which Pattern?" if purpose == "train" else "Run Simulation For Which Pattern?"
    draw_text(pt_text, font, WHITE, WIDTH // 2, HEIGHT // 2 - 150, center=True)
    draw_text(f"Press '1' for {PATTERN_NAMES[0]}", font, WHITE, WIDTH // 2, HEIGHT // 2 - 50, center=True)
    draw_text(f"Press '2' for {PATTERN_NAMES[1]}", font, WHITE, WIDTH // 2, HEIGHT // 2 + 0, center=True)
    draw_text(f"Press '3' for {PATTERN_NAMES[2]}", font, WHITE, WIDTH // 2, HEIGHT // 2 + 50, center=True)
    draw_text("Press ESC to return to previous menu", small_font, WHITE, WIDTH // 2, HEIGHT // 2 + 125, center=True); pygame.display.flip()

def draw_message(message, duration_ms=2000):
    # helper to display a temporary message (e.g., model saved)
    screen.fill(BLACK); draw_text(message, font, WHITE, WIDTH // 2, HEIGHT // 2, center=True)
    pygame.display.flip(); pygame.time.wait(duration_ms)

def draw_simulation(streetlamp, cars, hour, flow_rate, mode=None, pattern_type=None, is_training=False, training_run=0, motion_override=False):
    # main drawing function for the simulation screen
    screen.fill(BLACK); pygame.draw.rect(screen,DARK_GRAY,(0,HEIGHT//2+150,WIDTH,100)); pygame.draw.rect(screen,DARK_GREEN,(0,HEIGHT//2+250,WIDTH,HEIGHT//2-100)) # background
    streetlamp.draw(); [car.draw() for car in cars] # draw lamp and cars
    # draw text info overlay
    yo=10; draw_text(f"Time: {hour:.2f}/{SIMULATION_DURATION}h",font,WHITE,10,yo); yo+=30
    draw_text(streetlamp.get_energy_display(),font,WHITE,10,yo); yo+=30; draw_text(f"Current Flow: {int(flow_rate)} cars/hr",font,WHITE,10,yo); yo+=30
    draw_text(streetlamp.get_status_display(),font,WHITE,10,yo); yo+=30
    # display override message if active in mode 3
    if mode == MODE_ML_MOTION and motion_override:
        draw_text("Motion Detection Override Active", small_font, OVERRIDE_COLOR, 10, yo)
        yo += 25
    # display mode/pattern names top right
    if mode is not None and pattern_type is not None:
         if 0 <= pattern_type < len(PATTERN_NAMES) and 0 <= mode < len(MODE_NAMES):
            mode_name = MODE_NAMES[mode]; mode_text = font.render(f"Mode: {mode_name}", True, WHITE)
            pattern_text = font.render(f"Pattern: {PATTERN_NAMES[pattern_type]}", True, WHITE)
            screen.blit(mode_text,(WIDTH-mode_text.get_width()-10,10)); screen.blit(pattern_text,(WIDTH-pattern_text.get_width()-10,40))
    # display training run number if training
    if is_training: tt=font.render(f"Training Run: {training_run+1}/{total_training_runs}",True,WHITE); screen.blit(tt,(10,yo)); yo+=30
    # display completion message
    if hour >= SIMULATION_DURATION and not is_training: draw_text("Sim complete. Press any key.",font,WHITE,WIDTH//2,HEIGHT//2,center=True)
    draw_text("Press ESC to return",small_font,WHITE,WIDTH-160,HEIGHT-30); pygame.display.flip()

# Core Simulation and Training Logic
def run_simulation_logic(pattern_type, mode, predictor=None):
    # runs the main simulation loop for modes 0, 1, 2, 3
    cars = []; streetlamp = Streetlamp(WIDTH // 2, HEIGHT // 2 + 150)
    flow_monitor = TrafficFlowMonitor(); traffic_pattern_instance = TrafficPattern(pattern_type) # new pattern instance each run
    # deterministic time setup
    hour = 0.0; hour_delta_per_frame = (1.0 / FPS) / SECONDS_PER_HOUR if FPS > 0 and SECONDS_PER_HOUR > 0 else 0
    previous_hour = 0.0; last_prediction_time = -TIME_WINDOW
    current_flow_rate = 0; last_ml_prediction = STATE_OFF # store last prediction

    # initial lamp state
    if mode == MODE_ALWAYS_ON: streetlamp.set_state(STATE_FULL)
    else: streetlamp.set_state(STATE_OFF)

    running = True; clock = pygame.time.Clock(); sim_complete = False

    while running:
        motion_override_active = False # reset flag each frame

        for event in pygame.event.get(): # handle input
            if event.type == pygame.QUIT: return "QUIT"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "MENU"
                if sim_complete: running = False # end on key press after completion

        if not sim_complete: # only run simulation logic if not complete
            if hour >= SIMULATION_DURATION: # check completion
                sim_complete = True; hour = SIMULATION_DURATION
                streetlamp.calculate_energy(hour, previous_hour); # final energy calc
                previous_hour = hour
            else:
                # simulation steps
                target_flow = max(0, traffic_pattern_instance.get_target_flow(hour)) # get target flow
                # spawn cars based on probability derived from target flow
                frames_per_hour_sim = SECONDS_PER_HOUR * FPS
                spawn_probability = target_flow / frames_per_hour_sim if frames_per_hour_sim > 0 else 0
                if random.random() < spawn_probability: cars.append(Car()); flow_monitor.record_car()
                # move cars
                for car in cars: car.move()
                # remove off-screen cars
                cars = [car for car in cars if car.x > -car.length]
                # update flow rate (mainly for display)
                current_flow_rate = flow_monitor.update_flow_rate(hour)

                # determine lamp state based on mode
                final_state = STATE_OFF
                if mode == MODE_ALWAYS_ON: # mode 0
                    final_state = STATE_FULL
                elif mode == MODE_MOTION: # mode 1
                    car_in_range = any(abs((c.x + c.length / 2) - streetlamp.x) < streetlamp.detection_range for c in cars)
                    final_state = STATE_FULL if car_in_range else STATE_OFF
                elif mode == MODE_ML_ONLY or mode == MODE_ML_MOTION: # modes 2 or 3
                    # get ML prediction periodically
                    if predictor and predictor.is_trained and hour >= last_prediction_time + TIME_WINDOW:
                         last_ml_prediction = predictor.predict(hour)
                         last_prediction_time = hour
                    ml_state = last_ml_prediction

                    if mode == MODE_ML_ONLY: # mode 2
                        final_state = ml_state
                    elif mode == MODE_ML_MOTION: # mode 3
                        car_in_range = any(abs((c.x + c.length / 2) - streetlamp.x) < streetlamp.detection_range for c in cars)
                        if car_in_range:
                            final_state = max(ml_state, STATE_HALF) # ensure at least half brightness
                            if ml_state < STATE_HALF: # check if override actually changed state
                                motion_override_active = True # set flag for display
                        else:
                            final_state = ml_state # no motion, use ML state

                streetlamp.set_state(final_state) # apply final state
                streetlamp.calculate_energy(hour, previous_hour) # calculate energy for frame
                previous_hour = hour # store hour for next frame's calculation
                # increment time deterministically
                hour += hour_delta_per_frame
                hour = min(SIMULATION_DURATION, hour) # cap at duration

        # update flow rate display value even if paused
        if sim_complete: current_flow_rate = flow_monitor.update_flow_rate(hour)
        # draw screen, passing override status
        draw_simulation(streetlamp, cars, hour, current_flow_rate, mode, pattern_type, False, 0, motion_override_active)
        # control frame rate (slower when paused)
        clock.tick(10 if sim_complete else FPS)

    return "MENU" # return to menu after completion


def run_training_and_save(pattern_type):
    # runs multiple simulations, collects data, trains model, saves it
    master_collector = TrafficDataCollector(); clock = pygame.time.Clock()
    global total_training_runs
    # display starting message
    screen.fill(BLACK); draw_text(f"Starting Training for {PATTERN_NAMES[pattern_type]}...", font, WHITE, WIDTH // 2, HEIGHT // 2 - 20, center=True)
    draw_text(f"({total_training_runs} runs, please wait)", small_font, WHITE, WIDTH // 2, HEIGHT // 2 + 20, center=True); pygame.display.flip(); pygame.time.wait(1500)

    for run in range(total_training_runs): # main training loop
        print(f"--- Starting Training Run {run + 1}/{total_training_runs} for {PATTERN_NAMES[pattern_type]} ---")
        # setup for single run
        collector = TrafficDataCollector(); cars = []; streetlamp = Streetlamp(WIDTH // 2, HEIGHT // 2 + 150)
        flow_monitor = TrafficFlowMonitor(); traffic_pattern_instance = TrafficPattern(pattern_type)
        hour = 0.0; hour_delta_per_frame = (1.0 / FPS) / SECONDS_PER_HOUR if FPS > 0 and SECONDS_PER_HOUR > 0 else 0
        previous_hour = 0.0; last_collection_time = -TIME_WINDOW
        running_training_run = True; frame_count = 0; current_flow_rate = 0

        while running_training_run: # inner loop for one training run
            frame_count += 1
            for event in pygame.event.get(): # allow quit/escape
                if event.type == pygame.QUIT: return "QUIT"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: print("Training interrupted."); return "MENU"

            if hour >= SIMULATION_DURATION: # check end condition
                 hour = SIMULATION_DURATION; running_training_run = False
                 if hour >= last_collection_time + TIME_WINDOW: # collect final data point
                      current_flow_rate = flow_monitor.update_flow_rate(hour)
                      collector.collect_sample(hour, current_flow_rate)
                      last_collection_time = hour
            else: # run simulation steps for training
                 target_flow = max(0, traffic_pattern_instance.get_target_flow(hour))
                 frames_per_hour_sim = SECONDS_PER_HOUR * FPS
                 spawn_probability = target_flow / frames_per_hour_sim if frames_per_hour_sim > 0 else 0
                 if random.random() < spawn_probability: cars.append(Car()); flow_monitor.record_car()
                 for car in cars: car.move()
                 cars = [car for car in cars if car.x > -car.length]
                 current_flow_rate = flow_monitor.update_flow_rate(hour)
                 # collect data periodically
                 if hour >= last_collection_time + TIME_WINDOW:
                     collector.collect_sample(hour, current_flow_rate); last_collection_time = hour
                 # update visual state based on current flow (for display only)
                 csv = STATE_OFF
                 if current_flow_rate >= TFR_THRESHOLD_FULL: csv = STATE_FULL
                 elif current_flow_rate >= TFR_THRESHOLD_75: csv = STATE_75_PERCENT
                 elif current_flow_rate >= TFR_THRESHOLD_HALF: csv = STATE_HALF
                 streetlamp.set_state(csv)
                 streetlamp.calculate_energy(hour, previous_hour) # needed for display consistency
                 previous_hour = hour
                 hour += hour_delta_per_frame # increment time
                 hour = min(SIMULATION_DURATION, hour)

            # draw training screen (pass mode 2 for display purposes)
            draw_simulation(streetlamp, cars, hour, current_flow_rate, MODE_ML_ONLY, pattern_type, True, run)
            clock.tick(FPS) # maintain simulation speed

        master_collector.combine_data(collector) # add run data to master list
        print(f"--- Training Run {run + 1} complete. Samples: {len(collector.labels)} ---")

    # after all runs, train and save the final model
    print(f"--- All {total_training_runs} runs done. Total samples: {len(master_collector.labels)} ---")
    print("Now training the model..."); draw_text("Training complete. Now fitting model...", font, WHITE, WIDTH // 2, HEIGHT // 2, center=True); pygame.display.flip()
    predictor = StreetlightPredictor()
    training_successful = predictor.train(np.array(master_collector.time_data), np.array(master_collector.labels))
    if not training_successful: draw_message("Model training failed or skipped.", 3000); return "FAILED"
    # save the trained predictor (model + scaler)
    filepath = get_model_filepath(pattern_type)
    if filepath is None: draw_message("Error saving model (invalid pattern).", 3000); return "FAILED"
    try:
        os.makedirs(MODEL_DIR, exist_ok=True); # ensure directory exists
        data_to_save = {'model': predictor.model, 'scaler': predictor.scaler}
        joblib.dump(data_to_save, filepath); print(f"Model saved to {filepath}")
        draw_message(f"Model for {PATTERN_NAMES[pattern_type]} saved!", 2500); return "SAVED"
    except Exception as e: print(f"Error saving model: {e}"); draw_message(f"Error saving model: {e}", 4000); return "FAILED"


def load_predictor(pattern_type):
    # loads the saved model and scaler for a given pattern
    filepath = get_model_filepath(pattern_type)
    if filepath is None: draw_message("Cannot load model (invalid pattern).", 3000); return None
    print(f"Attempting to load model from: {filepath}")
    try:
        # load the dictionary containing model and scaler
        loaded_data = joblib.load(filepath); loaded_model = loaded_data['model']; loaded_scaler = loaded_data['scaler']
        # create predictor instance and load the data
        predictor = StreetlightPredictor(); predictor.load_model_data(loaded_model, loaded_scaler)
        print("Model and scaler loaded successfully."); return predictor
    except FileNotFoundError: print(f"Error: Model file not found at {filepath}. Train first."); draw_message(f"Model for {PATTERN_NAMES[pattern_type]} not found. Train first.", 3500); return None
    except Exception as e: print(f"Error loading model: {e}"); draw_message(f"Error loading model: {e}", 4000); return None

# Main Game Loop
def main():
    # manages game state and user input flow
    game_state = MENU; running = True; selected_pattern_type = None
    loaded_predictor = None; clock = pygame.time.Clock()
    selected_mode = None # track chosen mode

    while running:
        events = pygame.event.get()
        for event in events: # global event handling
            if event.type == pygame.QUIT: running = False; break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # handle ESC press based on current state
                    if game_state in [PATTERN_SELECT, ML_MENU]: game_state = MENU
                    elif game_state == ML_TRAIN_PATTERN_SELECT: game_state = ML_MENU
                    elif game_state == ML_RUN_PATTERN_SELECT:
                        # return to correct menu based on original mode selection
                        game_state = MENU if selected_mode == MODE_ML_MOTION else ML_MENU
                    elif game_state == MENU: running = False; break
        if not running: break # exit if QUIT detected

        # state machine logic
        if game_state == MENU:
            draw_menu()
            for event in events: # menu specific input
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_0: selected_mode = MODE_ALWAYS_ON; game_state = PATTERN_SELECT
                    elif event.key == pygame.K_1: selected_mode = MODE_MOTION; game_state = PATTERN_SELECT
                    elif event.key == pygame.K_2: selected_mode = MODE_ML_ONLY; game_state = ML_MENU
                    elif event.key == pygame.K_3: selected_mode = MODE_ML_MOTION; game_state = ML_RUN_PATTERN_SELECT # mode 3 goes direct to run select

        elif game_state == PATTERN_SELECT: # for modes 0 & 1
             draw_pattern_select(purpose="run")
             for event in events:
                 if event.type == pygame.KEYDOWN:
                    ps=-1 # pattern selection
                    if event.key==pygame.K_1: ps=MALL_ROAD
                    elif event.key==pygame.K_2: ps=NIGHT_ROAD
                    elif event.key==pygame.K_3: ps=OFFICE_ROAD
                    if ps!=-1:
                        spt=ps; print(f"\nSelected Mode: {MODE_NAMES[selected_mode]}, Pattern: {PATTERN_NAMES[spt]}")
                        sr=run_simulation_logic(spt,selected_mode,predictor=None) # run non-ML sim
                        if sr=="QUIT": running=False
                        else: game_state=MENU

        elif game_state == ML_MENU: # for mode 2 (ML Only) options
            draw_ml_menu()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: game_state = ML_TRAIN_PATTERN_SELECT # go train
                    elif event.key == pygame.K_2: game_state = ML_RUN_PATTERN_SELECT # go run (mode 2)

        elif game_state == ML_TRAIN_PATTERN_SELECT:
            draw_pattern_select(purpose="train")
            for event in events:
                if event.type == pygame.KEYDOWN:
                    ps=-1
                    if event.key==pygame.K_1: ps=MALL_ROAD
                    elif event.key==pygame.K_2: ps=NIGHT_ROAD
                    elif event.key==pygame.K_3: ps=OFFICE_ROAD
                    if ps!=-1: spt=ps; game_state=PERFORM_TRAINING # go to training state

        elif game_state == ML_RUN_PATTERN_SELECT: # for modes 2 & 3 run selection
            draw_pattern_select(purpose="run")
            for event in events:
                if event.type == pygame.KEYDOWN:
                    ps=-1
                    if event.key==pygame.K_1: ps=MALL_ROAD
                    elif event.key==pygame.K_2: ps=NIGHT_ROAD
                    elif event.key==pygame.K_3: ps=OFFICE_ROAD
                    if ps!=-1: spt=ps; game_state=PERFORM_LOADING # try loading model

        elif game_state == PERFORM_TRAINING:
            ts=run_training_and_save(spt) # run the training process
            if ts=="QUIT": running=False
            else: game_state=ML_MENU # return to ML menu afterwards

        elif game_state == PERFORM_LOADING:
            loaded_predictor = load_predictor(spt) # try loading the model
            if loaded_predictor is not None: game_state = SIMULATION # success -> run sim
            else: game_state = MENU if selected_mode == MODE_ML_MOTION else ML_MENU # fail -> return to appropriate menu

        elif game_state == SIMULATION: # runs mode 2 or 3 simulation
            print(f"Starting Simulation Run for Mode {selected_mode}...")
            sr=run_simulation_logic(spt, selected_mode, loaded_predictor) # run ML sim
            loaded_predictor=None # clear loaded predictor
            if sr=="QUIT": running=False
            else: game_state=MENU # return to main menu

        clock.tick(FPS) # limit frame rate

    pygame.quit(); print("Simulation exited.")

if __name__ == "__main__":
    # ensure model directory exists on startup
    try:
        if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR); print(f"Created directory: {MODEL_DIR}")
    except OSError as e: print(f"Error creating directory {MODEL_DIR}: {e}")
    main() # run the main function
