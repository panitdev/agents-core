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
    /// Runtime-only cancel signal; never serialized.
    #[serde(skip)]
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
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
            cancellation_token: None,
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

    pub fn with_cancellation_token(
        mut self,
        token: tokio_util::sync::CancellationToken,
    ) -> Self {
        self.cancellation_token = Some(token);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

enum Phase {
    Streaming,
    Completed(Vec<ToolCall>),
}

/// Folds a [`ChatStream`] into a committable assistant turn.
///
/// Normal path: poll events into `push`, then call `finish_complete` after `Done`.
/// Interrupted path: stop polling and call `finish_interrupted` to get a
/// text-only message with no tool_calls, safe to commit as-is.
pub struct StreamAccumulator {
    content: String,
    phase: Phase,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            phase: Phase::Streaming,
        }
    }

    /// Fold one stream event into the accumulator.
    ///
    /// `Done` finalizes the tool-call list. `ThinkingDelta` and
    /// `ToolCallArgumentDelta` are handled by history recording on the
    /// caller side; the accumulator ignores them.
    pub fn push(&mut self, event: ChatStreamEvent) {
        match event {
            ChatStreamEvent::ContentDelta(s) => self.content.push_str(&s),
            // `Done` is the sole source of finalized tool calls; partial
            // builder state from ToolCallArgumentDelta is not retained here.
            ChatStreamEvent::Done(tool_calls) => {
                self.phase = Phase::Completed(tool_calls);
            }
            ChatStreamEvent::ThinkingDelta(_) | ChatStreamEvent::ToolCallArgumentDelta { .. } => {}
        }
    }

    /// Consume the accumulator after a complete stream (phase == Completed).
    ///
    /// Returns the assistant message and the tool calls for dispatch.
    ///
    /// # Panics
    /// Panics if called before a `Done` event was pushed.
    pub fn finish_complete(self) -> (ChatMessage, Vec<ToolCall>) {
        let Phase::Completed(tool_calls) = self.phase else {
            panic!("finish_complete called before Done event");
        };
        let content = (!self.content.is_empty()).then_some(self.content);
        let message = ChatMessage::assistant(
            content,
            (!tool_calls.is_empty()).then(|| tool_calls.clone()),
        );
        (message, tool_calls)
    }

    /// Consume the accumulator on interrupt (phase == Streaming).
    ///
    /// Returns a text-only message, or `None` if nothing was accumulated.
    /// All tool-call builder state is discarded — partial arguments are
    /// never parseable, and no `tool_call` entry enters the message, so
    /// the next request requires no synthetic tool result.
    pub fn finish_interrupted(self) -> Option<ChatMessage> {
        if self.content.is_empty() {
            None
        } else {
            Some(ChatMessage::assistant(Some(self.content), None))
        }
    }
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ChatMessage, ChatStreamEvent, StreamAccumulator, ToolCall};

    #[test]
    fn normal_path_text_only() {
        let mut acc = StreamAccumulator::new();
        acc.push(ChatStreamEvent::ContentDelta("hello ".into()));
        acc.push(ChatStreamEvent::ContentDelta("world".into()));
        acc.push(ChatStreamEvent::Done(vec![]));

        let (msg, tool_calls) = acc.finish_complete();
        assert_eq!(msg, ChatMessage::assistant(Some("hello world".into()), None));
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn normal_path_with_tool_calls() {
        let mut acc = StreamAccumulator::new();
        acc.push(ChatStreamEvent::ContentDelta("running".into()));
        acc.push(ChatStreamEvent::ToolCallArgumentDelta {
            id: "c1".into(),
            name: "shell".into(),
            arguments_delta: "{\"cmd\":".into(),
        });
        acc.push(ChatStreamEvent::ToolCallArgumentDelta {
            id: "c1".into(),
            name: "shell".into(),
            arguments_delta: "\"ls\"}".into(),
        });
        let done_calls = vec![ToolCall {
            id: "c1".into(),
            name: "shell".into(),
            arguments: json!({"cmd": "ls"}),
        }];
        acc.push(ChatStreamEvent::Done(done_calls.clone()));

        let (msg, tool_calls) = acc.finish_complete();
        assert_eq!(
            msg,
            ChatMessage::assistant(Some("running".into()), Some(done_calls.clone()))
        );
        assert_eq!(tool_calls, done_calls);
    }

    #[test]
    fn interrupted_with_partial_text() {
        let mut acc = StreamAccumulator::new();
        acc.push(ChatStreamEvent::ContentDelta("partial".into()));
        // Never receives Done — simulates mid-stream interrupt.

        let msg = acc.finish_interrupted().expect("should produce a message");
        assert_eq!(msg, ChatMessage::assistant(Some("partial".into()), None));
        // Crucially, no tool_calls in the message.
        assert!(matches!(
            msg,
            ChatMessage::Assistant { tool_calls: None, .. }
        ));
    }

    #[test]
    fn interrupted_with_arg_deltas_but_no_text() {
        let mut acc = StreamAccumulator::new();
        acc.push(ChatStreamEvent::ToolCallArgumentDelta {
            id: "c1".into(),
            name: "shell".into(),
            arguments_delta: "{\"cmd\":".into(),
        });
        // Interrupted before Done — partial args discarded, no text accumulated.

        assert!(acc.finish_interrupted().is_none());
    }

    #[test]
    fn interrupted_empty() {
        let acc = StreamAccumulator::new();
        assert!(acc.finish_interrupted().is_none());
    }

    #[test]
    fn thinking_delta_ignored() {
        let mut acc = StreamAccumulator::new();
        acc.push(ChatStreamEvent::ThinkingDelta("step 1".into()));
        acc.push(ChatStreamEvent::ContentDelta("answer".into()));
        acc.push(ChatStreamEvent::Done(vec![]));

        let (msg, _) = acc.finish_complete();
        assert_eq!(msg, ChatMessage::assistant(Some("answer".into()), None));
    }
}
