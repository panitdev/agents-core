pub use chrono::{DateTime, Utc};
pub use uuid::Uuid;

pub type AgentId = Uuid;
pub type TaskId = Uuid;
pub type SubscriptionId = Uuid;

/// Runtime state of an agent.
///
/// # Variants
///
/// - `Starting`: Agent is initializing but not yet ready to handle requests
/// - `Running`: Agent is active and can process tasks
/// - `Stopping`: Agent is gracefully shutting down
/// - `Stopped`: Agent has completed shutdown and released resources
/// - `Failed`: Agent encountered an unrecoverable error
///
/// # Examples
///
/// ```
/// use panit_agents_core::AgentStatus;
///
/// let status = AgentStatus::Running;
/// assert_eq!(status.to_string(), "Running");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AgentStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
}

impl Default for AgentStatus {
    fn default() -> Self {
        Self::Starting
    }
}

impl std::fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentStatus::Starting => write!(f, "Starting"),
            AgentStatus::Running => write!(f, "Running"),
            AgentStatus::Stopping => write!(f, "Stopping"),
            AgentStatus::Stopped => write!(f, "Stopped"),
            AgentStatus::Failed => write!(f, "Failed"),
        }
    }
}
