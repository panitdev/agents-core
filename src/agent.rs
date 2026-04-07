use crate::ToolRegistry;
use crate::error::{AgentError, AgentResult};
use crate::llm::ChatMessage;
use crate::types::AgentId;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Type-erased agent stored in the registry.
///
/// Wraps an `Arc<dyn Agent<Delta = D> + Send + Sync>`. Accepts both concrete agent
/// types and pre-boxed `Arc<dyn Agent<..>>` via `From`/`Into` conversions.
pub struct DynAgent<D, S = ()>(pub Arc<dyn Agent<Delta = D, State = S> + Send + Sync>);

impl<D, S> Clone for DynAgent<D, S> {
    fn clone(&self) -> Self {
        DynAgent(self.0.clone())
    }
}

impl<D, S> std::ops::Deref for DynAgent<D, S> {
    type Target = dyn Agent<Delta = D, State = S> + Send + Sync;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Convert any concrete agent type into a `DynAgent`.
impl<D, S, T: Agent<Delta = D, State = S> + Send + Sync + 'static> From<T> for DynAgent<D, S> {
    fn from(agent: T) -> Self {
        DynAgent(Arc::new(agent))
    }
}

/// Convert a pre-boxed `Arc<dyn Agent<..>>` into a `DynAgent` without extra allocation.
impl<D, S> From<Arc<dyn Agent<Delta = D, State = S> + Send + Sync>> for DynAgent<D, S> {
    fn from(arc: Arc<dyn Agent<Delta = D, State = S> + Send + Sync>) -> Self {
        DynAgent(arc)
    }
}

pub type Sender<D> = tokio::sync::broadcast::Sender<D>;

/// Core trait for implementing agents that can process inputs and produce outputs.
///
/// Agents are the fundamental execution unit in the framework. Each agent has:
/// - A unique identity (`id`, `name`, `agent_type`)
/// - Optional system prompt for LLM context
/// - Typed input (serializable for cross-agent communication)
/// - Internal state accessible during execution
///
/// # Implementing
///
/// ```
/// use async_trait::async_trait;
/// use panit_agents_core::agent::{Agent, AgentConfig, AgentContext, AgentResult};
/// use panit_agents_core::types::AgentId;
/// use std::collections::HashMap;
///
/// struct MyAgent {
///     id: AgentId,
///     name: String,
///     state: HashMap<String, serde_json::Value>,
/// }
///
/// #[async_trait]
/// impl Agent for MyAgent {
///     type Input = serde_json::Value;
///     type State = HashMap<String, serde_json::Value>;
///
///     fn id(&self) -> AgentId { self.id }
///     fn name(&self) -> &str { &self.name }
///     fn agent_type(&self) -> &str { "my_agent" }
///
///     async fn execute(&self, input: Self::Input, _context: &AgentContext) -> AgentResult<Vec<panit_agents_core::llm::ChatMessage>> {
///         Ok(vec![panit_agents_core::llm::ChatMessage::user(input)])
///     }
///
///     fn state(&self) -> &Self::State { &self.state }
/// }
/// ```
#[async_trait]
pub trait Agent: Send + Sync {
    type Delta;
    type State;

    /// Unique identifier of the agent
    ///
    /// Must be unique on registry
    fn id(&self) -> &str;
    /// Display name for the agent
    ///
    /// This name is used for logging
    fn name(&self) -> &str;
    /// Description of the agent
    ///
    /// Agents use this to decide what agent to spawn
    fn description(&self) -> &str;

    fn system_prompt(&self) -> Option<&str> {
        None
    }

    /// Returns the names of tools this agent requires.
    ///
    /// Default implementation returns empty vec (no tools required).
    fn provided_tools(&self) -> Vec<String> {
        Vec::new()
    }

    async fn execute(
        &self,
        input: &str,
        context: &AgentContext<Self::Delta, Self::State>,
        tx: Sender<Self::Delta>,
    ) -> AgentResult<Vec<ChatMessage>>;
}

/// Execution context provided to agents at runtime.
///
/// Contains references to shared resources that agents use during execution:
/// - Registry for discovering and invoking other agents
/// - Tool registry for executing available tools
/// - Optional LLM client for language model interactions
///
/// # Default
///
/// Returns a default context with nil agent ID and empty registries.
/// This is suitable for testing but should not be used in production
/// where proper registry and tool discovery is required.
///
/// # Examples
///
/// ```
/// use panit_agents_core::agent::AgentContext;
///
/// let context = AgentContext::default();
/// assert_eq!(context.agent_id, uuid::Uuid::nil());
/// ```
#[derive(Clone)]
pub struct AgentContext<D, S = ()> {
    pub id: String,
    pub agents: AgentRegistry<D, S>,
    pub tools: ToolRegistry<AgentContext<D, S>>,
    pub llm_client: Option<Arc<dyn LLMClient>>,
    pub history: Vec<ChatMessage>,
    pub state: S,
    pub cancel_token: CancellationToken,
}

impl<D: 'static, S> Default for AgentContext<D, S>
where
    S: Default,
{
    fn default() -> Self {
        Self {
            id: String::new(),
            agents: AgentRegistry::new(),
            tools: ToolRegistry::new(),
            llm_client: None,
            history: Vec::new(),
            state: S::default(),
            cancel_token: CancellationToken::new(),
        }
    }
}

impl<D: 'static, S> AgentContext<D, S>
where
    S: Default,
{
    pub fn with_id(id: String) -> AgentContext<D, S> {
        Self {
            id,
            ..Self::default()
        }
    }

    /// Build a [`ChatRequest`] with the given message appended to the history.
    ///
    /// The caller is responsible for updating `history` after receiving the LLM response.
    pub fn build_request(&self, message: ChatMessage) -> ChatRequest {
        let mut messages = self.history.clone();
        messages.push(message);
        ChatRequest::new(messages)
    }
}

/// Configuration for agent initialization and execution.
///
/// # Default
///
/// - `max_retries`: 3
/// - `timeout_ms`: 30000 (30 seconds)
/// - `system_prompt`: None
/// - `metadata`: empty
///
/// # Examples
///
/// ```
/// use panit_agents_core::agent::AgentConfig;
///
/// let config = AgentConfig::default();
/// assert_eq!(config.max_retries, 3);
/// assert_eq!(config.timeout_ms, 30000);
///
/// let custom = AgentConfig {
///     max_retries: 5,
///     timeout_ms: 60000,
///     system_prompt: Some("You are a helpful assistant".to_string()),
///     metadata: std::collections::HashMap::new(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_retries: u32,
    pub timeout_ms: u64,
    pub system_prompt: Option<String>,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            timeout_ms: 30000,
            system_prompt: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

pub struct AgentRegistry<D, S = ()> {
    agents: Arc<HashMap<String, DynAgent<D, S>>>,
}

impl<D, S> Clone for AgentRegistry<D, S> {
    fn clone(&self) -> Self {
        Self {
            agents: self.agents.clone(),
        }
    }
}

impl<D, S> AgentRegistry<D, S> {
    pub fn new() -> AgentRegistry<D, S> {
        Self {
            agents: Arc::new(HashMap::new()),
        }
    }

    pub fn builder() -> AgentRegistryBuilder<D, S> {
        AgentRegistryBuilder::new()
    }

    pub fn get(&self, id: &str) -> Option<&DynAgent<D, S>> {
        self.agents.get(id)
    }

    pub fn all(&self) -> Vec<&DynAgent<D, S>> {
        self.agents.values().collect()
    }
}

impl<D> Default for AgentRegistry<D> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AgentRegistryBuilder<D, S = ()> {
    agents: HashMap<String, DynAgent<D, S>>,
}

impl<D, S> AgentRegistryBuilder<D, S> {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    pub fn register(mut self, tool: impl Into<DynAgent<D, S>>) -> Self {
        let tool = tool.into();
        self.agents.insert(tool.id().to_string(), tool);
        self
    }

    pub fn build(self) -> AgentRegistry<D, S> {
        AgentRegistry::<D, S> {
            agents: Arc::new(self.agents),
        }
    }
}

impl<D, S> Default for AgentRegistryBuilder<D, S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a registered agent.
///
/// Returned by `AgentRegistry::list_all()` to provide agent information
/// without needing to construct the full agent instance.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub id: AgentId,
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<Capability>,
    pub status: crate::types::AgentStatus,
}

/// Describes a capability that an agent can provide.
///
/// Capabilities enable agent discovery and matching. Each capability has:
/// - A unique name for identification
/// - Human-readable description
/// - JSON schemas for input/output validation
///
/// # Examples
///
/// ```
/// use panit_agents_core::agent::Capability;
/// use serde_json::json;
///
/// let capability = Capability {
///     name: "text_generation".to_string(),
///     description: "Generates text from prompts".to_string(),
///     input_schema: json!({ "type": "object", "properties": { "prompt": { "type": "string" } } }),
///     output_schema: json!({ "type": "string" }),
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Capability {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
}

impl Default for Capability {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            input_schema: serde_json::Value::Null,
            output_schema: serde_json::Value::Null,
        }
    }
}

/// Client for interacting with language models.
///
/// Provides a unified interface for LLM interactions, supporting both
/// plain text and JSON-structured responses. Implementations may wrap
/// various LLM APIs (OpenAI, Anthropic, local models, etc.).
///
/// # Examples
///
/// ```
/// use panit_agents_core::agent::LLMClient;
/// use panit_agents_core::MockLLMClient;
///
/// async fn example() -> panit_agents_core::AgentResult<String> {
///     let client = MockLLMClient::new("gpt-4".to_string());
///     client.complete("Hello, world!").await
/// }
/// ```
use crate::llm::{ChatRequest, ChatResponse, ChatStream};

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> AgentResult<ChatResponse>;

    async fn chat_stream(
        &self,
        request: ChatRequest,
        cancel_token: CancellationToken,
    ) -> AgentResult<ChatStream>;

    async fn complete(&self, prompt: &str) -> AgentResult<String> {
        let request = ChatRequest::new(vec![crate::llm::ChatMessage::user(prompt)]);
        let response = self.chat(request).await?;
        if let crate::llm::ChatMessage::Assistant {
            content: Some(content),
            ..
        } = response.message
        {
            Ok(content)
        } else {
            Err(AgentError::ExecutionError(
                "Expected assistant message with content".to_string(),
            ))
        }
    }
}
