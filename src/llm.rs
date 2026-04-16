use crate::error::{LLMError, LLMResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;

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
/// async fn example() -> panit_agents_core::LLMResult<String> {
///     let client = MockLLMClient::new("gpt-4".to_string());
///     client.complete("Hello, world!").await
/// }
/// ```
#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> LLMResult<ChatResponse>;

    async fn chat_stream(&self, request: ChatRequest) -> LLMResult<ChatStream>;

    async fn complete(&self, prompt: &str) -> LLMResult<String> {
        let request = ChatRequest::new(vec![crate::llm::ChatMessage::user(prompt)]);
        let response = self.chat(request).await?;
        if let crate::llm::ChatMessage::Assistant {
            content: Some(content),
            ..
        } = response.message
        {
            Ok(content)
        } else {
            Err(LLMError::ParseError(
                "Expected assistant message with content".to_string(),
            ))
        }
    }
}

/// Reasoning configuration for models that support extended thinking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Reasoning effort level (e.g., "low", "medium", "high").
    pub effort: Option<String>,
    /// Whether to include reasoning tokens in the response.
    pub enabled: Option<bool>,
    /// Maximum number of tokens for reasoning.
    pub max_tokens: Option<u32>,
}

impl ReasoningConfig {
    pub fn new() -> Self {
        Self {
            effort: None,
            enabled: None,
            max_tokens: None,
        }
    }

    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        self.effort = Some(effort.into());
        self
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = Some(enabled);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub temperature: Option<f32>,
    pub model: Option<String>,
    pub reasoning: Option<ReasoningConfig>,
    pub fallbacks: Option<Vec<String>>,
    pub authorization: Option<String>,
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            tools: None,
            temperature: None,
            model: None,
            reasoning: None,
            fallbacks: None,
            authorization: None,
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn with_fallbacks(mut self, fallbacks: Vec<String>) -> Self {
        self.fallbacks = Some(fallbacks);
        self
    }

    pub fn with_authorization(mut self, authorization: String) -> Self {
        self.authorization = Some(authorization);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        tool_call_id: Option<String>,
        content: String,
    },
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage::User {
            content: content.into(),
        }
    }

    pub fn assistant(
        content: impl Into<Option<String>>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        ChatMessage::Assistant {
            content: content.into(),
            tool_calls,
        }
    }

    pub fn assistant_empty() -> Self {
        ChatMessage::Assistant {
            content: None,
            tool_calls: None,
        }
    }

    pub fn tool(tool_call_id: impl Into<Option<String>>, content: impl Into<String>) -> Self {
        ChatMessage::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }

    pub fn content(&self) -> Option<&str> {
        match self {
            ChatMessage::System { content } => Some(content.as_str()),
            ChatMessage::User { content } => Some(content.as_str()),
            ChatMessage::Assistant { content, .. } => content.as_deref(),
            ChatMessage::Tool { content, .. } => Some(content.as_str()),
        }
    }

    pub fn role(&self) -> &'static str {
        match self {
            ChatMessage::System { .. } => "system",
            ChatMessage::User { .. } => "user",
            ChatMessage::Assistant { .. } => "assistant",
            ChatMessage::Tool { .. } => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatResponse {
    pub fn new(message: ChatMessage) -> Self {
        Self {
            message,
            tool_calls: None,
        }
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }
}

/// An event yielded by a streaming chat response.
///
/// Content deltas are emitted immediately as they arrive. Tool call
/// argument deltas are emitted as fragments arrive, and the full list
/// of tool calls is delivered once the stream ends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatStreamEvent {
    /// Incremental reasoning/thinking text (e.g. from DeepSeek-R1 or extended-thinking models).
    ThinkingDelta(String),
    /// Incremental text content from the assistant.
    ContentDelta(String),
    /// Incremental tool call argument fragment.
    ///
    /// Emitted as each chunk of tool call arguments arrives from the stream.
    /// The `arguments_delta` field contains only the new fragment (not the
    /// accumulated total). Consumers should accumulate these if they need
    /// the full arguments before the tool call completes.
    ToolCallArgumentDelta {
        /// Unique tool call ID (stable across deltas for the same call).
        id: String,
        /// Tool/function name (may be empty until the name fragment arrives).
        name: String,
        /// Incremental argument fragment (JSON text fragment).
        arguments_delta: String,
    },
    /// Stream finished. Contains all tool calls requested by the model (may be empty).
    Done(Vec<ToolCall>),
}

/// Stream of chat response events.
pub type ChatStream =
    Pin<Box<dyn futures_util::Stream<Item = Result<ChatStreamEvent, LLMError>> + Send>>;

/// Mock LLM client for testing and development.
///
/// Returns predefined responses without making actual API calls.
/// Supports custom response templates for testing specific scenarios.
///
/// # Examples
///
/// ```
/// use panit_agents_core::llm::MockLLMClient;
/// use panit_agents_core::agent::LLMClient;
/// use serde_json::json;
///
/// async fn example() -> panit_agents_core::LLMResult<String> {
///     let client = MockLLMClient::new("gpt-4".to_string());
///     client.complete("Hello").await
/// }
///
/// async fn with_template() -> panit_agents_core::LLMResult<String> {
///     let client = MockLLMClient::new("gpt-4".to_string())
///         .with_response_template(json!("Mock response"));
///     client.complete("Hello").await
/// }
/// ```
pub struct MockLLMClient {
    model: String,
    response_template: Option<Value>,
}

impl MockLLMClient {
    pub fn new(model: String) -> Self {
        Self {
            model,
            response_template: None,
        }
    }

    pub fn with_response_template(mut self, template: Value) -> Self {
        self.response_template = Some(template);
        self
    }
}

#[async_trait]
impl LLMClient for MockLLMClient {
    async fn chat(&self, request: ChatRequest) -> LLMResult<ChatResponse> {
        tracing::info!(
            "MockLLMClient chat with {} messages",
            request.messages.len()
        );

        let last_message = request.messages.last();
        let prompt = match last_message {
            Some(ChatMessage::User { content }) => content.clone(),
            Some(ChatMessage::Tool { content, .. }) => content.clone(),
            _ => String::new(),
        };

        if let Some(ref template) = self.response_template {
            let response_str = serde_json::to_string(template)
                .map_err(|e| LLMError::ParseError(e.to_string()))?;
            return Ok(ChatResponse::new(ChatMessage::assistant(
                Some(response_str),
                None,
            )));
        }

        Ok(ChatResponse::new(ChatMessage::assistant(
            Some(format!(
                "Mock response for: {}",
                &prompt[..prompt.len().min(50)]
            )),
            None,
        )))
    }

    async fn chat_stream(&self, request: ChatRequest) -> LLMResult<ChatStream> {
        let response = self.chat(request).await?;
        let content = match response.message {
            ChatMessage::Assistant {
                content: Some(c), ..
            } => c,
            _ => String::new(),
        };

        let stream = futures_util::stream::iter(vec![
            Ok(ChatStreamEvent::ContentDelta(content)),
            Ok(ChatStreamEvent::Done(vec![])),
        ]);

        Ok(Box::pin(stream) as ChatStream)
    }
}
