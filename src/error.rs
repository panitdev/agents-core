use thiserror::Error;
use uuid::Uuid;

/// Result type returned by agent operations.
///
/// # Examples
///
/// ```
/// use panit_agents_core::AgentResult;
///
/// async fn example() -> AgentResult<String> {
///     Ok("success".to_string())
/// }
/// ```
pub type AgentResult<T> = Result<T, AgentError>;

/// Errors that can occur during agent execution.
///
/// # Variants
///
/// - `InitializationError`: Agent failed to initialize with the given configuration
/// - `ExecutionError`: Agent execution failed (e.g., LLM call failed, tool execution failed)
/// - `TimeoutError`: Operation exceeded its configured timeout
/// - `ResourceExhaustedError`: Agent hit a resource limit (e.g., rate limit, token limit)
/// - `AgentNotFoundError`: Requested agent does not exist in the registry
/// - `CapabilityMismatchError`: Agent does not support the required capability
/// - `SerializationError`: Failed to serialize or deserialize data
/// - `TaskNotFoundError`: Requested task does not exist
/// - `RegistryError`: Registry operation failed
/// - `ToolError`: Tool execution failed
///
/// # Examples
///
/// ```
/// use panit_agents_core::AgentError;
///
/// let err = AgentError::AgentNotFoundError(uuid::Uuid::nil());
/// assert!(err.to_string().contains("not found"));
/// ```
#[derive(Error, Debug, Clone)]
pub enum AgentError {
    #[error("initialization: {0}")]
    InitializationError(String),
    #[error("execution: {0}")]
    ExecutionError(String),
    #[error("timed out")]
    TimeoutError,
    #[error("resource exhausted error")]
    ResourceExhaustedError,
    #[error("agent not found: {0}")]
    AgentNotFoundError(Uuid),
    #[error("capability mismatch")]
    CapabilityMismatchError,
    #[error("serialization: {0}")]
    SerializationError(String),
    #[error("task not found: {0}")]
    TaskNotFoundError(Uuid),
    #[error("registry: {0}")]
    RegistryError(String),
    #[error("tool: {0}")]
    ToolError(String),
    #[error("no client supplied")]
    NoClientSupplied,
}
