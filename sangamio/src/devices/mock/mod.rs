//! Mock device driver for hardware-free robot simulation
//!
//! This module provides a complete simulation of CRL-200S robot hardware,
//! enabling SLAM and navigation algorithm development without physical robots.
//!
//! # Overview
//!
//! The mock driver simulates all sensors and actuators of the CRL-200S:
//!
//! | Component | Simulation Method |
//! |-----------|-------------------|
//! | Lidar (Delta-2D) | Ray-casting against occupancy grid |
//! | Wheel encoders | Differential drive kinematics + slip noise |
//! | IMU (gyro/accel) | Physics-based with configurable noise |
//! | Bumpers | Collision detection against map |
//! | Cliff sensors | Cliff mask lookup or boundary detection |
//! | Battery/buttons | Fixed configurable states |
//!
//! # Map Format
//!
//! Uses ROS-standard PGM + YAML format for maximum compatibility:
//!
//! ```yaml
//! # maps/example.yaml
//! image: example.pgm        # Grayscale occupancy grid
//! resolution: 0.02          # meters per pixel
//! origin: [-5.0, -5.0, 0.0] # [x, y, yaw] of bottom-left pixel
//! occupied_thresh: 0.65     # Darker = occupied
//! cliff_mask: cliffs.pgm    # Optional cliff layer
//! ```
//!
//! PGM pixel values:
//! - **White (255)**: Free space
//! - **Black (0)**: Occupied/wall
//! - **Gray (205)**: Unknown
//!
//! # Configuration
//!
//! Enable the `mock` feature to use this driver:
//!
//! ```bash
//! cargo build --features mock
//! cargo run --features mock -- mock.toml
//! ```
//!
//! Example configuration (`mock.toml`):
//!
//! ```toml
//! [device]
//! type = "mock"
//! name = "Mock CRL-200S"
//!
//! [device.simulation]
//! map_file = "maps/example.yaml"
//! start_x = 1.5
//! start_y = 3.5
//! start_theta = 0.0
//! speed_factor = 1.0    # 2.0 = 2x speed
//! random_seed = 42      # 0 = random each run
//!
//! [network]
//! bind_address = "0.0.0.0:5555"
//! ```
//!
//! # Simulation Loop
//!
//! The simulation runs at 110Hz (matching CRL-200S hardware):
//!
//! ```text
//! Every ~9ms:
//! 1. Read velocity commands from shared state
//! 2. Update physics (pose, collision detection)
//! 3. Generate encoder ticks from wheel velocities
//! 4. Generate IMU readings (gyro, accel, tilt)
//! 5. Check bumpers and cliff sensors
//! 6. Publish sensor_status via streaming channel
//!
//! Every ~200ms (5Hz):
//! 7. Generate lidar scan via ray-casting
//! 8. Update lidar sensor group
//! ```
//!
//! # Speed Factor
//!
//! The `speed_factor` setting accelerates simulation time:
//!
//! | Factor | sensor_status | lidar | Use Case |
//! |--------|---------------|-------|----------|
//! | 1.0 | 110 Hz | 5 Hz | Real-time testing |
//! | 2.0 | 220 Hz | 10 Hz | Faster algorithm iteration |
//! | 5.0 | 550 Hz | 25 Hz | Quick integration tests |
//!
//! # Noise Models
//!
//! All sensors include configurable noise for realistic testing:
//!
//! - **Lidar**: Range stddev, angle stddev, miss rate, quality decay
//! - **Encoders**: Wheel slip (multiplicative), quantization noise
//! - **IMU**: Per-axis stddev, bias, gyro drift rate
//!
//! # Thread Model
//!
//! ```text
//! ┌─────────────────┐
//! │   Main Thread   │
//! │  (initialize)   │
//! └────────┬────────┘
//!          │ spawns
//!          ▼
//! ┌─────────────────┐     ┌─────────────────┐
//! │ Simulation Loop │────▶│ Streaming Chan  │
//! │  (mock-sim)     │     │ (sensor_status) │
//! └─────────────────┘     └─────────────────┘
//! ```
//!
//! # Module Structure
//!
//! - [`config`]: Configuration structures for all simulation parameters
//! - [`physics`]: Differential drive kinematics and collision handling
//! - [`lidar_sim`]: Ray-casting lidar simulation
//! - [`imu_sim`]: IMU data generation with noise
//! - [`encoder_sim`]: Wheel encoder simulation with slip
//! - [`sensor_sim`]: Bumper, cliff, and binary sensor simulation
//! - [`map_loader`]: PGM + YAML map loading
//! - [`noise`]: Configurable noise generator

pub mod config;
pub mod encoder_sim;
pub mod imu_sim;
pub mod lidar_sim;
pub mod map_loader;
pub mod noise;
pub mod physics;
pub mod sensor_sim;

use crate::config::DeviceConfig;
use crate::core::driver::{DeviceDriver, DriverInitResult};
use crate::core::types::{Command, Kinematics, SensorGroupData, SensorValue};
use crate::error::{Error, Result};

use config::{DockFaultConfig, SimulationConfig};
use encoder_sim::EncoderSimulator;
use imu_sim::ImuSimulator;
use lidar_sim::LidarSimulator;
use map_loader::SimulationMap;
use noise::NoiseGenerator;
use physics::PhysicsState;
use sensor_sim::{ActuatorState, BumperSimulator, CliffSimulator};

use crossbeam_channel::Sender;
use std::collections::HashMap;
use std::net::UdpSocket;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum MockProfile {
    Generic,
    RoborockS5Max,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ExternalPose {
    x: f32,
    y: f32,
    yaw: f32,
    linear_velocity: f32,
    angular_velocity: f32,
    bumper_left: bool,
    bumper_right: bool,
}

fn parse_external_pose(packet: &[u8]) -> Option<ExternalPose> {
    let text = std::str::from_utf8(packet).ok()?.trim();
    let values: Vec<&str> = text.split(',').collect();
    if values.len() != 7 {
        return None;
    }
    Some(ExternalPose {
        x: values[0].parse().ok()?,
        y: values[1].parse().ok()?,
        yaw: values[2].parse().ok()?,
        linear_velocity: values[3].parse().ok()?,
        angular_velocity: values[4].parse().ok()?,
        bumper_left: values[5] == "1",
        bumper_right: values[6] == "1",
    })
}

fn wrap_angle(angle: f32) -> f32 {
    let mut value = angle % std::f32::consts::TAU;
    if value >= std::f32::consts::PI {
        value -= std::f32::consts::TAU;
    } else if value < -std::f32::consts::PI {
        value += std::f32::consts::TAU;
    }
    value
}

fn dock_observation(config: &SimulationConfig, x: f32, y: f32, yaw: f32) -> (u16, u16, bool) {
    let dock = &config.dock;
    let dx = dock.x - x;
    let dy = dock.y - y;
    let forward = dx * dock.yaw.cos() + dy * dock.yaw.sin();
    let lateral = -dx * dock.yaw.sin() + dy * dock.yaw.cos();
    let sight_bearing = wrap_angle(dy.atan2(dx) - yaw);
    let heading_error = wrap_angle(dock.yaw - yaw);
    let guidance = wrap_angle(heading_error + lateral.atan2(forward.max(0.40)));
    let visible = dx.hypot(dy) <= dock.signal_range && sight_bearing.abs() <= dock.signal_angle;
    let masks = if !visible {
        (0, 0)
    } else if guidance > dock.center_angle {
        (0x1a, 0x18)
    } else if guidance < -dock.center_angle {
        (0, 0x15)
    } else {
        (0x15, 0x18)
    };
    let contact = (forward - dock.contact_distance).abs() <= 0.025
        && lateral.abs() <= dock.contact_lateral
        && wrap_angle(yaw - dock.yaw).abs() <= dock.contact_yaw;
    (masks.0, masks.1, contact)
}

fn apply_s5max_scenario(config: &mut SimulationConfig, scenario: &str) -> Result<()> {
    config.dock_faults = DockFaultConfig::default();
    match scenario {
        "centered" => {}
        "lateral-offset" => {
            config.start_y = config.dock.y + 0.12;
            config.start_theta = config.dock.yaw;
        }
        "heading-error" => {
            config.start_y = config.dock.y;
            config.start_theta = config.dock.yaw + 0.12;
        }
        "beacon-loss" => {
            config.dock_faults.beacon_blackout_after_sec = Some(2.0);
            config.dock_faults.beacon_blackout_duration_sec = 1.5;
        }
        "missed-contact" => {
            config.dock_faults.missed_contact_attempts = 1;
        }
        "delayed-charging" => {
            config.dock_faults.charging_delay_sec = 2.5;
        }
        _ => {
            return Err(Error::Config(format!(
                "unknown S5 Max mock scenario: {scenario}"
            )));
        }
    }
    config.scenario = scenario.to_string();
    Ok(())
}

#[derive(Default)]
struct DockFaultState {
    previous_geometric_contact: bool,
    contact_attempts: u32,
    accepted_contact: bool,
    charging_since: Option<Instant>,
}

impl DockFaultState {
    fn update(
        &mut self,
        faults: &DockFaultConfig,
        geometric_contact: bool,
        charger_enabled: bool,
        now: Instant,
    ) -> (bool, bool) {
        if geometric_contact && !self.previous_geometric_contact {
            self.contact_attempts += 1;
            self.accepted_contact = self.contact_attempts > faults.missed_contact_attempts;
        } else if !geometric_contact {
            self.accepted_contact = false;
        }
        self.previous_geometric_contact = geometric_contact;

        let contact = geometric_contact && self.accepted_contact;
        if contact && charger_enabled {
            let started = self.charging_since.get_or_insert(now);
            let charging =
                now.duration_since(*started).as_secs_f32() >= faults.charging_delay_sec.max(0.0);
            (contact, charging)
        } else {
            self.charging_since = None;
            (contact, false)
        }
    }
}

fn signal_blacked_out(
    faults: &DockFaultConfig,
    motion_started_at: Option<Instant>,
    now: Instant,
) -> bool {
    let (Some(after), Some(started)) = (faults.beacon_blackout_after_sec, motion_started_at) else {
        return false;
    };
    let elapsed = now.duration_since(started).as_secs_f32();
    elapsed >= after && elapsed < after + faults.beacon_blackout_duration_sec.max(0.0)
}

/// Atomic f32 wrapper using AtomicU32
struct AtomicF32(AtomicU32);

impl AtomicF32 {
    fn new(val: f32) -> Self {
        Self(AtomicU32::new(val.to_bits()))
    }

    fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.0.load(order))
    }

    fn store(&self, val: f32, order: Ordering) {
        self.0.store(val.to_bits(), order);
    }
}

/// Component state for tracking enable/disable commands
/// Mirrors CRL-200S behavior where lidar and motors must be explicitly enabled
pub struct ComponentState {
    /// Whether lidar motor is enabled (must be enabled to get scan data)
    pub lidar_enabled: AtomicBool,
    /// Whether wheel motors are enabled (must be enabled for drive commands to work)
    pub wheel_motor_enabled: AtomicBool,
}

impl ComponentState {
    fn new() -> Self {
        Self {
            lidar_enabled: AtomicBool::new(false),
            wheel_motor_enabled: AtomicBool::new(false),
        }
    }
}

/// Shared state for thread communication
struct SharedState {
    linear_vel: AtomicF32,
    angular_vel: AtomicF32,
    shutdown: AtomicBool,
    components: ComponentState,
}

impl SharedState {
    fn new() -> Self {
        Self {
            linear_vel: AtomicF32::new(0.0),
            angular_vel: AtomicF32::new(0.0),
            shutdown: AtomicBool::new(false),
            components: ComponentState::new(),
        }
    }
}

/// Mock device driver for simulating CRL-200S hardware
pub struct MockDriver {
    config: SimulationConfig,
    shared_state: Arc<SharedState>,
    simulation_handle: Option<JoinHandle<()>>,
    actuator_state: Arc<Mutex<ActuatorState>>,
    profile: MockProfile,
}

impl MockDriver {
    /// Create a new mock driver from device configuration
    pub fn new(device_config: DeviceConfig) -> Result<Self> {
        let profile = if device_config.device_type == "roborock_s5max_mock" {
            MockProfile::RoborockS5Max
        } else {
            MockProfile::Generic
        };
        let mut sim_config = device_config.simulation.ok_or_else(|| {
            Error::Config("Mock device requires simulation configuration".to_string())
        })?;
        if profile == MockProfile::RoborockS5Max {
            sim_config.robot.wheel_base = 0.229;
            sim_config.robot.ticks_per_meter = 1.0 / 0.00079799996;
            sim_config.robot.max_linear_speed = 0.30;
            sim_config.robot.max_angular_speed = 1.60;
            sim_config.robot.robot_radius = 0.1765;
            let scenario = std::env::var("SANGAMIO_MOCK_SCENARIO")
                .unwrap_or_else(|_| sim_config.scenario.clone());
            apply_s5max_scenario(&mut sim_config, &scenario)?;
            log::info!("S5 Max mock scenario: {scenario}");
        }

        Ok(Self {
            config: sim_config,
            shared_state: Arc::new(SharedState::new()),
            simulation_handle: None,
            actuator_state: Arc::new(Mutex::new(ActuatorState::new())),
            profile,
        })
    }

    fn shutdown_all(&mut self) {
        self.shared_state.shutdown.store(true, Ordering::Relaxed);

        if let Some(handle) = self.simulation_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod s5max_mock_tests {
    use super::*;

    fn config() -> SimulationConfig {
        SimulationConfig {
            dock: config::DockConfig {
                x: 1.0,
                y: 0.0,
                yaw: 0.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn dock_signal_reports_both_receivers_on_center_line() {
        let (front, right, contact) = dock_observation(&config(), 0.5, 0.0, 0.0);
        assert_eq!((front, right), (0x15, 0x18));
        assert!(!contact);
    }

    #[test]
    fn dock_signal_keeps_the_center_pattern_near_the_dock() {
        let (front, right, contact) = dock_observation(&config(), 0.7, 0.0, 0.0);
        assert_eq!((front, right), (0x15, 0x18));
        assert!(!contact);
    }

    #[test]
    fn dock_signal_uses_overlapping_front_and_right_receivers() {
        let (front, right, _) = dock_observation(&config(), 0.5, -0.2, 0.0);
        assert_eq!(front, 0x1a);
        assert_eq!(right, 0x18);
    }

    #[test]
    fn dock_contact_requires_position_and_heading_alignment() {
        let (_, _, contact) = dock_observation(&config(), 0.82, 0.0, 0.0);
        assert!(contact);
        let (_, _, lateral_miss) = dock_observation(&config(), 0.82, 0.04, 0.0);
        assert!(!lateral_miss);
        let (_, _, yaw_miss) = dock_observation(&config(), 0.82, 0.0, 0.2);
        assert!(!yaw_miss);
    }

    #[test]
    fn scenario_presets_are_deterministic() {
        let mut lateral = config();
        apply_s5max_scenario(&mut lateral, "lateral-offset").unwrap();
        assert_eq!(lateral.start_y, 0.12);

        let mut delayed = config();
        apply_s5max_scenario(&mut delayed, "delayed-charging").unwrap();
        assert_eq!(delayed.dock_faults.charging_delay_sec, 2.5);
    }

    #[test]
    fn external_pose_packet_preserves_gazebo_observation() {
        let pose = parse_external_pose(b"2.1,1.4,-0.2,0.03,-0.1,1,0").unwrap();

        assert_eq!(pose.x, 2.1);
        assert_eq!(pose.y, 1.4);
        assert_eq!(pose.yaw, -0.2);
        assert_eq!(pose.linear_velocity, 0.03);
        assert_eq!(pose.angular_velocity, -0.1);
        assert!(pose.bumper_left);
        assert!(!pose.bumper_right);
    }

    #[test]
    fn missed_contact_accepts_the_next_attempt() {
        let faults = DockFaultConfig {
            missed_contact_attempts: 1,
            ..Default::default()
        };
        let now = Instant::now();
        let mut state = DockFaultState::default();
        assert_eq!(state.update(&faults, true, false, now), (false, false));
        state.update(&faults, false, false, now);
        assert_eq!(state.update(&faults, true, false, now), (true, false));
    }

    #[test]
    fn charging_delay_uses_stable_contact_time() {
        let faults = DockFaultConfig {
            charging_delay_sec: 2.5,
            ..Default::default()
        };
        let now = Instant::now();
        let mut state = DockFaultState::default();
        assert_eq!(state.update(&faults, true, true, now), (true, false));
        assert_eq!(
            state.update(&faults, true, true, now + Duration::from_secs(3)),
            (true, true)
        );
    }
}

impl DeviceDriver for MockDriver {
    fn initialize(&mut self) -> Result<DriverInitResult> {
        log::info!("Initializing mock device driver");

        // Load map
        let map = SimulationMap::load(&self.config.map_file)?;
        log::info!(
            "Loaded map: {}x{} pixels, resolution: {} m/px",
            map.width(),
            map.height(),
            map.resolution()
        );

        // Initialize physics state
        let physics = PhysicsState::new(
            self.config.start_x,
            self.config.start_y,
            self.config.start_theta,
            &self.config.robot,
        );

        // Create noise generator
        let noise = NoiseGenerator::new(self.config.random_seed);

        // Create sensor groups
        let sensor_status = Arc::new(Mutex::new(SensorGroupData::new("sensor_status")));
        let lidar_data = Arc::new(Mutex::new(SensorGroupData::new("lidar")));
        let version_data = Arc::new(Mutex::new(SensorGroupData::new("device_version")));
        let dock_ir = Arc::new(Mutex::new(SensorGroupData::new("dock_ir")));

        // Set version info
        {
            let mut ver = version_data
                .lock()
                .map_err(|e| Error::MutexPoisoned(format!("version_data lock failed: {}", e)))?;
            ver.set(
                "version_string",
                SensorValue::String(if self.profile == MockProfile::RoborockS5Max {
                    "Roborock S5 Max Mock v1.0".to_string()
                } else {
                    "MockDevice v1.0".to_string()
                }),
            );
            ver.set("version_code", SensorValue::I32(100));
            ver.touch();
        }

        // Create streaming channel for sensor_status
        let (stream_tx, stream_rx) = crate::core::types::create_stream_channel();

        // Clone references for simulation thread
        let sim_config = self.config.clone();
        let shared = Arc::clone(&self.shared_state);
        let sensor_status_clone = Arc::clone(&sensor_status);
        let lidar_data_clone = Arc::clone(&lidar_data);
        let actuator_state_clone = Arc::clone(&self.actuator_state);
        let dock_ir_clone = Arc::clone(&dock_ir);
        let profile = self.profile;

        // Spawn simulation thread
        let handle = thread::Builder::new()
            .name("mock-simulation".to_string())
            .spawn(move || {
                simulation_loop(
                    sim_config,
                    shared,
                    map,
                    physics,
                    noise,
                    sensor_status_clone,
                    lidar_data_clone,
                    stream_tx,
                    actuator_state_clone,
                    dock_ir_clone,
                    profile,
                );
            })
            .map_err(|e| Error::Other(format!("Failed to spawn simulation thread: {}", e)))?;

        self.simulation_handle = Some(handle);

        log::info!("Mock device driver initialized");

        // Build result
        let mut sensor_data = HashMap::new();
        sensor_data.insert("sensor_status".to_string(), sensor_status);
        sensor_data.insert("lidar".to_string(), lidar_data);
        sensor_data.insert("device_version".to_string(), version_data);
        if self.profile == MockProfile::RoborockS5Max {
            sensor_data.insert("dock_ir".to_string(), dock_ir);
        }
        let kinematics = Kinematics {
            wheel_m_per_tick: if self.profile == MockProfile::RoborockS5Max {
                0.00079799996
            } else {
                1.0 / self.config.robot.ticks_per_meter
            },
            wheel_track_m: if self.profile == MockProfile::RoborockS5Max {
                0.229
            } else {
                self.config.robot.wheel_base
            },
            wheel_tick_bits: 16,
            gyro_rad_per_lsb: 1.0 / imu_sim::GYRO_SCALE,
            accel_mps2_per_lsb: Some(9.81 / imu_sim::ACCEL_SCALE),
        };
        sensor_data.insert(
            "kinematics".to_string(),
            Arc::new(Mutex::new(kinematics.to_sensor_group())),
        );

        let mut stream_receivers = HashMap::new();
        stream_receivers.insert("sensor_status".to_string(), stream_rx);

        Ok(DriverInitResult {
            sensor_data,
            stream_receivers,
        })
    }

    fn send_command(&mut self, cmd: Command) -> Result<()> {
        match cmd {
            Command::ComponentControl { id, action } => {
                match id.as_str() {
                    "drive" => {
                        // Handle drive commands (matches CRL-200S behavior)
                        // - Enable: enables wheel motors (required before Configure works)
                        // - Configure: sets velocity (only if wheel motors enabled)
                        // - Disable: stops and disables wheel motors
                        match &action {
                            crate::core::types::ComponentAction::Enable { .. } => {
                                self.shared_state
                                    .components
                                    .wheel_motor_enabled
                                    .store(true, Ordering::Relaxed);
                                log::debug!("Drive enabled (wheel motors on)");
                            }
                            crate::core::types::ComponentAction::Configure { config } => {
                                if self.profile != MockProfile::RoborockS5Max
                                    && !self
                                        .shared_state
                                        .components
                                        .wheel_motor_enabled
                                        .load(Ordering::Relaxed)
                                {
                                    log::warn!(
                                        "Drive command ignored: wheel motors not enabled (send Enable first)"
                                    );
                                    return Ok(());
                                }
                                if let Some(SensorValue::F32(v)) = config.get("linear") {
                                    self.shared_state.linear_vel.store(*v, Ordering::Relaxed);
                                }
                                if let Some(SensorValue::F32(w)) = config.get("angular") {
                                    self.shared_state.angular_vel.store(*w, Ordering::Relaxed);
                                }
                                log::debug!(
                                    "Drive command: linear={}, angular={}",
                                    self.shared_state.linear_vel.load(Ordering::Relaxed),
                                    self.shared_state.angular_vel.load(Ordering::Relaxed)
                                );
                            }
                            crate::core::types::ComponentAction::Disable { .. } => {
                                self.shared_state
                                    .components
                                    .wheel_motor_enabled
                                    .store(false, Ordering::Relaxed);
                                self.shared_state.linear_vel.store(0.0, Ordering::Relaxed);
                                self.shared_state.angular_vel.store(0.0, Ordering::Relaxed);
                                log::debug!("Drive disabled (wheel motors off)");
                            }
                            _ => {}
                        }
                    }
                    "lidar" => {
                        // Handle lidar enable/disable
                        match &action {
                            crate::core::types::ComponentAction::Enable { .. } => {
                                self.shared_state
                                    .components
                                    .lidar_enabled
                                    .store(true, Ordering::Relaxed);
                                log::debug!("Lidar enabled");
                            }
                            crate::core::types::ComponentAction::Disable { .. } => {
                                self.shared_state
                                    .components
                                    .lidar_enabled
                                    .store(false, Ordering::Relaxed);
                                log::debug!("Lidar disabled");
                            }
                            crate::core::types::ComponentAction::Configure { config } => {
                                // Configure also enables lidar (like CRL-200S)
                                self.shared_state
                                    .components
                                    .lidar_enabled
                                    .store(true, Ordering::Relaxed);
                                if let Some(SensorValue::U8(pwm)) = config.get("pwm") {
                                    log::debug!("Lidar enabled (PWM={}%)", pwm);
                                } else {
                                    log::debug!("Lidar enabled via configure");
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {
                        // Handle other actuator commands (vacuum, brushes, etc.)
                        if let Ok(mut state) = self.actuator_state.lock() {
                            state.handle_command(
                                &id,
                                &action,
                                self.config.actuators.log_state_changes,
                            );
                        }
                    }
                }
            }
            Command::Shutdown => {
                log::info!("Shutdown command received");
                self.shutdown_all();
            }
            Command::ProtocolSync => {
                // Nothing to do for mock
            }
        }
        Ok(())
    }
}

impl Drop for MockDriver {
    fn drop(&mut self) {
        self.shutdown_all();
    }
}

/// Main simulation loop
fn simulation_loop(
    config: SimulationConfig,
    shared: Arc<SharedState>,
    map: SimulationMap,
    mut physics: PhysicsState,
    noise: NoiseGenerator,
    sensor_status: Arc<Mutex<SensorGroupData>>,
    lidar_data: Arc<Mutex<SensorGroupData>>,
    stream_tx: Sender<SensorGroupData>,
    _actuator_state: Arc<Mutex<ActuatorState>>,
    dock_ir: Arc<Mutex<SensorGroupData>>,
    profile: MockProfile,
) {
    // Create simulators
    let mut lidar_sim = LidarSimulator::new(&config.lidar, noise.clone());
    let mut imu_sim = ImuSimulator::new(&config.imu, noise.clone());
    let mut encoder_sim =
        EncoderSimulator::new(&config.encoder, config.robot.ticks_per_meter, noise.clone());
    let bumper_sim = BumperSimulator::new(&config.bumpers);
    let cliff_sim = CliffSimulator::new(&config.cliffs);

    // Timing
    let base_interval_us = (1_000_000.0 / 110.0) as u64; // ~9090us for 110Hz
    let scaled_interval =
        Duration::from_micros((base_interval_us as f32 / config.speed_factor) as u64);
    let lidar_interval_ticks = (110.0 / config.lidar.scan_rate_hz) as u32; // ~22 ticks at 5Hz

    let mut last_time = Instant::now();
    let mut lidar_tick_counter: u32 = 0;
    let mut motion_started_at = None;
    let mut dock_fault_state = DockFaultState::default();
    let external_pose_socket = std::env::var("SANGAMIO_MOCK_POSE_UDP")
        .ok()
        .and_then(|address| match UdpSocket::bind(&address) {
            Ok(socket) => {
                if let Err(error) = socket.set_nonblocking(true) {
                    log::error!("Cannot make the external pose socket nonblocking: {error}");
                    return None;
                }
                log::info!("External Gazebo pose input is active on {address}");
                Some(socket)
            }
            Err(error) => {
                log::error!("Cannot bind the external pose socket at {address}: {error}");
                None
            }
        });
    let mut external_pose = None;
    let mut external_pose_buffer = [0_u8; 256];

    log::info!(
        "Simulation loop started: speed_factor={}, interval={:?}",
        config.speed_factor,
        scaled_interval
    );

    while !shared.shutdown.load(Ordering::Relaxed) {
        let loop_start = Instant::now();

        // Calculate delta time
        let now = Instant::now();
        let wall_dt = now.duration_since(last_time).as_secs_f32();
        let sim_dt = wall_dt * config.speed_factor;
        last_time = now;

        // Read velocity commands
        let linear_vel = shared.linear_vel.load(Ordering::Relaxed);
        let angular_vel = shared.angular_vel.load(Ordering::Relaxed);
        if motion_started_at.is_none() && (linear_vel.abs() > 1e-4 || angular_vel.abs() > 1e-4) {
            motion_started_at = Some(now);
        }

        if let Some(socket) = &external_pose_socket {
            while let Ok(size) = socket.recv(&mut external_pose_buffer) {
                if let Some(pose) = parse_external_pose(&external_pose_buffer[..size]) {
                    external_pose = Some(pose);
                }
            }
        }

        // Use Gazebo as the physics source when it supplies an external pose.
        let (observed_linear_vel, observed_angular_vel, collision) = if let Some(pose) =
            external_pose
        {
            physics.set_pose(pose.x, pose.y, pose.yaw);
            (pose.linear_velocity, pose.angular_velocity, false)
        } else {
            let collision = physics.update(sim_dt, linear_vel, angular_vel, &map, &config.robot);
            (linear_vel, angular_vel, collision)
        };
        if collision && config.log_level != "minimal" {
            log::debug!("Collision at ({:.3}, {:.3})", physics.x(), physics.y());
        }

        // Generate encoder data
        let (left_vel, right_vel) = physics.wheel_velocities(
            observed_linear_vel,
            observed_angular_vel,
            config.robot.wheel_base,
        );
        let (wheel_left, wheel_right) = encoder_sim.update(left_vel, right_vel, sim_dt);

        // Generate IMU data
        let imu_reading = imu_sim.generate(observed_angular_vel, observed_linear_vel, sim_dt);

        // Check bumpers and cliffs
        let (bumper_left, bumper_right) = if let Some(pose) = external_pose {
            (pose.bumper_left, pose.bumper_right)
        } else {
            bumper_sim.check(
                &map,
                physics.x(),
                physics.y(),
                physics.theta(),
                config.robot.robot_radius,
            )
        };
        let (mut dock_front_mask, mut dock_right_mask, geometric_dock_contact) =
            if profile == MockProfile::RoborockS5Max {
                dock_observation(&config, physics.x(), physics.y(), physics.theta())
            } else {
                (0, 0, config.sensors.dock_connected)
            };
        if profile == MockProfile::RoborockS5Max
            && signal_blacked_out(&config.dock_faults, motion_started_at, now)
        {
            dock_front_mask = 0;
            dock_right_mask = 0;
        }
        let charger_enabled = _actuator_state
            .lock()
            .map(|state| state.is_enabled("charger"))
            .unwrap_or(false);
        let (dock_contact, dock_charging) = if profile == MockProfile::RoborockS5Max {
            dock_fault_state.update(
                &config.dock_faults,
                geometric_dock_contact,
                charger_enabled,
                now,
            )
        } else {
            (geometric_dock_contact, config.sensors.battery.is_charging)
        };
        let cliff_states = cliff_sim.check(&map, physics.x(), physics.y(), physics.theta());

        // Build sensor_status
        if let Ok(mut status) = sensor_status.lock() {
            status.set("wheel_left", SensorValue::U16(wheel_left));
            status.set("wheel_right", SensorValue::U16(wheel_right));

            status.set("gyro_x", SensorValue::I16(imu_reading.gyro[0]));
            status.set("gyro_y", SensorValue::I16(imu_reading.gyro[1]));
            status.set("gyro_z", SensorValue::I16(imu_reading.gyro[2]));
            status.set("accel_x", SensorValue::I16(imu_reading.accel[0]));
            status.set("accel_y", SensorValue::I16(imu_reading.accel[1]));
            status.set("accel_z", SensorValue::I16(imu_reading.accel[2]));
            status.set("tilt_x", SensorValue::I16(imu_reading.tilt[0]));
            status.set("tilt_y", SensorValue::I16(imu_reading.tilt[1]));
            status.set("tilt_z", SensorValue::I16(imu_reading.tilt[2]));

            status.set(
                "bumper_left",
                SensorValue::Bool(bumper_left || dock_contact),
            );
            status.set(
                "bumper_right",
                SensorValue::Bool(bumper_right || dock_contact),
            );

            // Cliff sensors
            status.set(
                "cliff_left_side",
                SensorValue::Bool(*cliff_states.get("left_side").unwrap_or(&false)),
            );
            status.set(
                "cliff_left_front",
                SensorValue::Bool(*cliff_states.get("left_front").unwrap_or(&false)),
            );
            status.set(
                "cliff_right_front",
                SensorValue::Bool(*cliff_states.get("right_front").unwrap_or(&false)),
            );
            status.set(
                "cliff_right_side",
                SensorValue::Bool(*cliff_states.get("right_side").unwrap_or(&false)),
            );

            // Battery (fixed)
            status.set(
                "battery_voltage",
                SensorValue::F32(config.sensors.battery.voltage),
            );
            status.set(
                "battery_level",
                SensorValue::U8(config.sensors.battery.level),
            );
            status.set(
                "is_charging",
                SensorValue::Bool(if profile == MockProfile::RoborockS5Max {
                    dock_charging
                } else {
                    config.sensors.battery.is_charging
                }),
            );

            // Other sensors (fixed)
            status.set(
                "dustbox_attached",
                SensorValue::Bool(config.sensors.dustbox_attached),
            );
            status.set(
                "is_dock_connected",
                SensorValue::Bool(if profile == MockProfile::RoborockS5Max {
                    dock_contact
                } else {
                    config.sensors.dock_connected
                }),
            );
            status.set(
                "start_button",
                SensorValue::U16(config.sensors.start_button),
            );
            status.set("dock_button", SensorValue::U16(config.sensors.dock_button));

            status.touch();

            // Send to streaming channel
            let _ = stream_tx.try_send(status.clone());
        }

        if profile == MockProfile::RoborockS5Max
            && let Ok(mut signal) = dock_ir.lock()
        {
            signal.set("front_mask", SensorValue::U16(dock_front_mask));
            signal.set("right_mask", SensorValue::U16(dock_right_mask));
            signal.touch();
        }

        // Generate lidar scan periodically (only if lidar is enabled)
        lidar_tick_counter += 1;
        if lidar_tick_counter >= lidar_interval_ticks {
            lidar_tick_counter = 0;

            // Only generate scan if lidar is enabled
            if shared.components.lidar_enabled.load(Ordering::Relaxed) {
                let scan = lidar_sim.generate_scan(&map, physics.x(), physics.y(), physics.theta());

                if let Ok(mut lidar) = lidar_data.lock() {
                    log::trace!(
                        "Generated lidar scan: {} points at ({:.2}, {:.2}, {:.2}°)",
                        scan.len(),
                        physics.x(),
                        physics.y(),
                        physics.theta().to_degrees()
                    );
                    lidar.set("scan", SensorValue::PointCloud2D(scan));
                    lidar.touch();
                }
            }
        }

        // Sleep for remaining interval
        let elapsed = loop_start.elapsed();
        if elapsed < scaled_interval {
            thread::sleep(scaled_interval - elapsed);
        }
    }

    log::info!("Simulation loop terminated");
}
