//! OpenAI-compatible LLM client implementation.
//!
//! This module provides an [`OpenAILLMClient`] that works with OpenAI's API
//! as well as OpenAI-compatible APIs (OpenRouter, local models, Azure OpenAI, etc.).
//!
//! # Examples
//!
//! ```
//! use panit_agents_core::providers::OpenAILLMClient;
//! use panit_agents_core::agent::LLMClient;
//!
//! async fn example() -> panit_agents_core::AgentResult<String> {
//!     let client = OpenAILLMClient::new("sk-api-key".to_string(), "gpt-4".to_string());
//!     client.complete("Hello, world!").await
//! }
//! ```

use crate::agent::LLMClient;
use crate::error::{AgentError, AgentResult};
use crate::llm::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, ToolCall, ToolDefinition,
};
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::{debug, info};

/// Configuration for the OpenAI-compatible LLM client.
///
/// # Examples
///
/// ```
/// use panit_agents_core::providers::OpenAIClientConfig;
///
/// let config = OpenAIClientConfig::new("sk-api-key".to_string(), "gpt-4".to_string())
///     .with_base_url("https://api.openai.com/v1");
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIClientConfig {
    /// API key for authentication.
    pub api_key: String,
    /// The model to use (e.g., "gpt-4", "gpt-3.5-turbo").
    pub model: String,
    /// Base URL for the API. Defaults to OpenAI's API endpoint.
    pub base_url: String,
    /// Optional organization ID for OpenAI API.
    pub organization: Option<String>,
    /// Optional project ID for OpenAI API.
    pub project: Option<String>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0.0 to 2.0).
    pub temperature: Option<f32>,
    /// Maximum number of tools to call.
    pub max_tools: Option<u32>,
    /// Whether to stream responses.
    pub stream: bool,
    /// Reasoning effort level for reasoning models (e.g., "low", "medium", "high").
    pub reasoning_effort: Option<String>,
    /// Fallback models for OpenRouter. When set, the `models` array is sent
    /// with the primary model first, followed by fallbacks in priority order.
    pub fallbacks: Option<Vec<String>>,
}

impl OpenAIClientConfig {
    /// Create a new configuration with the given API key and model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
            project: None,
            max_tokens: None,
            temperature: None,
            max_tools: None,
            stream: false,
            reasoning_effort: None,
            fallbacks: None,
        }
    }

    /// Set a custom base URL for OpenAI-compatible APIs.
    ///
    /// Common base URLs:
    /// - OpenAI: `https://api.openai.com/v1`
    /// - OpenRouter: `https://openrouter.ai/api/v1`
    /// - Local: `http://localhost:1234/v1`
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the organization ID.
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Set the project ID.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set the maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tools to call.
    pub fn with_max_tools(mut self, max_tools: u32) -> Self {
        self.max_tools = Some(max_tools);
        self
    }

    /// Enable response streaming.
    pub fn with_stream(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Set the reasoning effort level for reasoning models (e.g., "low", "medium", "high").
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Set fallback models for OpenRouter. When set, the `models` array is sent
    /// with the primary model first, followed by fallbacks in priority order.
    pub fn with_fallbacks(mut self, fallbacks: Vec<String>) -> Self {
        self.fallbacks = Some(fallbacks);
        self
    }
}

/// OpenAI-compatible LLM client.
///
/// Implements the [`LLMClient`] trait to provide chat completion capabilities
/// with support for OpenAI and OpenAI-compatible APIs.
///
/// # Features
///
/// - OpenAI API compatibility
/// - OpenAI-compatible APIs (OpenRouter, local models, etc.)
/// - Function/tool calling support
/// - Configurable base URL for self-hosted models
/// - Organization and project support for OpenAI
///
/// # Examples
///
/// ## Basic usage
///
/// ```
/// use panit_agents_core::providers::OpenAILLMClient;
/// use panit_agents_core::agent::LLMClient;
///
/// async fn example() -> panit_agents_core::AgentResult<String> {
///     let client = OpenAILLMClient::new("sk-api-key".to_string(), "gpt-4".to_string());
///     client.complete("What is the capital of France?").await
/// }
/// ```
///
/// ## With custom configuration
///
/// ```
/// use panit_agents_core::providers::{OpenAIClientConfig, OpenAILLMClient};
/// use panit_agents_core::agent::LLMClient;
/// use panit_agents_core::llm::{ChatMessage, ChatRequest};
///
/// async fn with_config() -> panit_agents_core::AgentResult<String> {
///     let config = OpenAIClientConfig::new("sk-api-key".to_string(), "gpt-4".to_string())
///         .with_base_url("https://openrouter.ai/api/v1")
///         .with_temperature(0.5)
///         .with_max_tokens(1000);
///
///     let client = OpenAILLMClient::with_config(config);
///     client.complete("Hello!").await
/// }
/// ```
///
/// ## Using OpenRouter
///
/// ```
/// use panit_agents_core::providers::{OpenAIClientConfig, OpenAILLMClient};
/// use panit_agents_core::agent::LLMClient;
///
/// async fn openrouter() -> panit_agents_core::AgentResult<String> {
///     let config = OpenAIClientConfig::new("sk-or-v2-...", "anthropic/claude-3.5-sonnet")
///         .with_base_url("https://openrouter.ai/api/v1");
///
///     let client = OpenAILLMClient::with_config(config);
///     client.complete("Hello!").await
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OpenAILLMClient {
    config: OpenAIClientConfig,
    http_client: Client,
}

impl OpenAILLMClient {
    /// Create a new OpenAI client with the given API key and model.
    ///
    /// Uses the default OpenAI API endpoint.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_config(OpenAIClientConfig::new(api_key, model))
    }

    /// Create a new OpenAI client with the given configuration.
    pub fn with_config(config: OpenAIClientConfig) -> Self {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            http_client,
        }
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.http_client = client;
        self
    }

    /// Get the configured model name.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Get the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Build the headers for API requests.
    fn build_headers(&self, request: &ChatRequest) -> HeaderMap {
        let mut headers = HeaderMap::new();
        let auth_value = request
            .authorization
            .clone()
            .unwrap_or_else(|| self.config.api_key.clone());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", auth_value)).unwrap(),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(ref org) = self.config.organization {
            headers.insert(
                HeaderName::from_static("openai-organization"),
                HeaderValue::from_str(org).unwrap(),
            );
        }

        if let Some(ref project) = self.config.project {
            headers.insert(
                HeaderName::from_static("openai-project"),
                HeaderValue::from_str(project).unwrap(),
            );
        }

        headers
    }

    /// Convert internal ChatMessage to OpenAI API format.
    fn to_openai_message(message: &ChatMessage) -> OpenAIMessage {
        match message {
            ChatMessage::System { content } => OpenAIMessage {
                role: "system".to_string(),
                content: content.clone(),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage::User { content } => OpenAIMessage {
                role: "user".to_string(),
                content: content.clone(),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage::Assistant {
                content,
                tool_calls,
            } => OpenAIMessage {
                role: "assistant".to_string(),
                content: content.clone().unwrap_or_default(),
                tool_call_id: None,
                tool_calls: tool_calls
                    .clone()
                    .map(|tool_calls| tool_calls.iter().map(Self::to_openai_tool_call).collect()),
            },
            ChatMessage::Tool {
                content,
                tool_call_id,
            } => OpenAIMessage {
                role: "tool".to_string(),
                content: content.clone(),
                tool_call_id: tool_call_id.clone(),
                tool_calls: None,
            },
        }
    }

    /// Convert ToolDefinition to OpenAI function format.
    fn to_openai_function(tool: &ToolDefinition) -> OpenAIFunction {
        OpenAIFunction {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        }
    }

    fn to_openai_tool_call(tool: &ToolCall) -> OpenAIToolCall {
        OpenAIToolCall {
            id: tool.id.clone(),
            tool_type: String::from("function"),
            function: OpenAIFunctionCall {
                name: tool.name.clone(),
                arguments: serde_json::to_string(&tool.arguments).unwrap_or_default(),
            },
        }
    }

    /// Build the API request body.
    fn build_request(&self, request: &ChatRequest) -> OpenAIChatRequest {
        let messages: Vec<OpenAIMessage> = request
            .messages
            .iter()
            .map(Self::to_openai_message)
            .collect();

        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| OpenAIFunctionDefinition {
                    r#type: "function".to_string(),
                    function: Self::to_openai_function(t),
                })
                .collect()
        });

        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());

        let fallbacks = request
            .fallbacks
            .clone()
            .or_else(|| self.config.fallbacks.clone());

        let models = fallbacks.map(|mut fb| {
            fb.insert(0, model.clone());
            fb
        });

        OpenAIChatRequest {
            model,
            messages,
            tools,
            temperature: request.temperature.or(self.config.temperature),
            max_tokens: self.config.max_tokens,
            stream: Some(self.config.stream),
            reasoning_effort: request
                .reasoning
                .as_ref()
                .and_then(|r| r.effort.clone())
                .or_else(|| self.config.reasoning_effort.clone()),
            models,
        }
    }

    /// Handle the API response and convert to ChatResponse.
    fn handle_response(&self, response: OpenAIChatResponse) -> AgentResult<ChatResponse> {
        let message = response
            .choices
            .first()
            .ok_or_else(|| AgentError::ExecutionError("No choices in response".to_string()))?;

        tracing::debug!("Parsed message: {:?}", message.message);

        let content = message
            .message
            .content
            .as_ref()
            .cloned()
            .unwrap_or_default();

        let chat_message = if content.is_empty() && message.message.tool_calls.is_some() {
            ChatMessage::Assistant {
                content: None,
                tool_calls: None,
            }
        } else {
            ChatMessage::Assistant {
                content: Some(content),
                tool_calls: None,
            }
        };

        let tool_calls = message.message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|call| {
                    let args = serde_json::from_str(call.function.arguments.as_ref())
                        .unwrap_or(serde_json::Value::Object(Default::default()));

                    ToolCall {
                        id: call.id.clone(),
                        name: call.function.name.clone(),
                        arguments: args,
                    }
                })
                .collect()
        });

        Ok(ChatResponse::new(chat_message).with_tool_calls(tool_calls.unwrap_or_default()))
    }

    /// Execute a chat request against the API.
    async fn execute_chat(&self, request: ChatRequest) -> AgentResult<ChatResponse> {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );

        let body = self.build_request(&request);
        let model = body.model.clone();

        tracing::debug!("Sending chat request to {}", url);
        tracing::debug!("Request body: {:?}", serde_json::to_string(&body).ok());

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers(&request))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentError::ExecutionError(format!(
                    "HTTP request failed: {}\nURL: {}\nModel: {}",
                    e, url, model
                ))
            })?;

        let status = response.status();

        if !status.is_success() {
            let response_body = response.text().await.unwrap_or_default();
            tracing::error!("API error: {} - {}", status, response_body);
            return Err(AgentError::ExecutionError(format!(
                "API error {}\nURL: {}\nModel: {}\nResponse: {}",
                status, url, model, response_body
            )));
        }

        let response_body = response.text().await.unwrap_or_default();

        tracing::info!(
            "Raw LLM response ({} chars): {}",
            response_body.len(),
            response_body
        );

        tracing::debug!("Response body: {}", response_body);

        let chat_response: OpenAIChatResponse =
            serde_json::from_str(&response_body).map_err(|e| {
                AgentError::ExecutionError(format!(
                    "Failed to parse OpenAI response: {}\nURL: {}\nModel: {}\nResponse body: {}",
                    e, url, model, response_body
                ))
            })?;

        tracing::debug!("Parsed chat_response: {:?}", chat_response);

        self.handle_response(chat_response)
    }

    /// Execute a streaming chat request against the API.
    async fn execute_chat_streaming(&self, request: ChatRequest) -> AgentResult<ChatStream> {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );

        let mut body = self.build_request(&request);
        body.stream = Some(true);
        let model = body.model.clone();

        debug!("send streaming chat request to {}", url);
        debug!(
            "body(stripped tool defs): {body:#?}",
            body = {
                let mut body = body.clone();
                body.tools = None;
                body
            }
        );

        let response = self
            .http_client
            .post(&url)
            .headers(self.build_headers(&request))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentError::ExecutionError(format!(
                    "HTTP request failed: {}\nURL: {}\nModel: {}",
                    e, url, model
                ))
            })?;

        let status = response.status();

        if !status.is_success() {
            let response_body = response.text().await.unwrap_or_default();
            tracing::error!("API error: {} - {}", status, response_body);
            return Err(AgentError::ExecutionError(format!(
                "API error {}\nURL: {}\nModel: {}\nResponse: {}",
                status, url, model, response_body
            )));
        }

        let stream = response.bytes_stream();
        let stream = stream.map(|r| {
            let re = r.as_ref().ok().and_then(|v| std::str::from_utf8(v).ok());
            if let Some(re) = re {
                if let Some(stripped) = re.strip_prefix("data: ")
                    && let Ok(mut parsed) = serde_json::from_str::<serde_json::Value>(stripped)
                {
                    if let Some(v) = parsed.as_object_mut() {
                        v.remove("created");
                        v.remove("id");
                        v.remove("object");
                        v.remove("provider");
                    }

                    // debug!("JSON from stream: {parsed:#?}");
                } else {
                    // debug!("raw from stream: {re:?}");
                }
            }
            r
        });

        Ok(Box::pin(SseStream::new(stream)) as ChatStream)
    }
}

#[async_trait]
impl LLMClient for OpenAILLMClient {
    async fn chat(&self, request: ChatRequest) -> AgentResult<ChatResponse> {
        self.execute_chat(request).await
    }

    async fn chat_stream(&self, request: ChatRequest) -> AgentResult<ChatStream> {
        self.execute_chat_streaming(request).await
    }
}

/// Message format for OpenAI API.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

/// Tool call in OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunctionCall,
}

/// Function call in OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

/// Function definition for tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Function definition wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIFunctionDefinition {
    #[serde(rename = "type")]
    r#type: String,
    function: OpenAIFunction,
}

/// Chat completion request for OpenAI API.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAIFunctionDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    /// OpenRouter fallback models array. When present, the API tries each
    /// model in order on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    models: Option<Vec<String>>,
}

/// Chat completion response from OpenAI API.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

/// Choice in OpenAI response.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIChoice {
    index: u32,
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

/// Response message from OpenAI.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIResponseMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

/// Usage statistics from OpenAI.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Streaming chunk from OpenAI SSE response.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIChunkResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIChunkChoice {
    index: u32,
    delta: OpenAIChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIChunkDelta {
    content: Option<String>,
    /// Reasoning/thinking tokens exposed by OpenRouter for extended-thinking models
    /// (e.g. DeepSeek-R1, Claude with thinking enabled).
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

/// Tool call delta in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIStreamToolCall {
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default)]
    tool_type: Option<String>,
    function: OpenAIStreamFunctionCall,
}

/// Function call delta in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIStreamFunctionCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Accumulates fragments of a single tool call during streaming.
#[derive(Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

/// Assemble completed tool calls from accumulated builders.
fn assemble_done(
    builders: &mut std::collections::HashMap<u32, ToolCallBuilder>,
) -> ChatStreamEvent {
    let mut entries: Vec<_> = builders.drain().collect();
    entries.sort_by_key(|(idx, _)| *idx);
    let tool_calls = entries
        .into_iter()
        .map(|(_, b)| {
            let arguments = serde_json::from_str(&b.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            ToolCall {
                id: b.id,
                name: b.name,
                arguments,
            }
        })
        .collect();
    ChatStreamEvent::Done(tool_calls)
}

/// Stream that parses SSE from OpenAI and yields [`ChatStreamEvent`]s.
///
/// Content deltas are forwarded immediately. Tool call fragments are accumulated
/// and delivered as a complete list in the final [`ChatStreamEvent::Done`] event.
struct SseStream<S> {
    inner: S,
    buffer: String,
    /// Builders keyed by tool-call index — accumulate fragments across chunks.
    tool_call_builders: std::collections::HashMap<u32, ToolCallBuilder>,
    /// Set to true once the underlying byte stream signals end-of-stream.
    stream_ended: bool,
    /// Set to true once `Done` has been emitted.
    done_sent: bool,
}

impl<S> SseStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
            tool_call_builders: std::collections::HashMap::new(),
            stream_ended: false,
            done_sent: false,
        }
    }
}

impl<S> Stream for SseStream<S>
where
    S: futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<ChatStreamEvent, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;

        loop {
            while let Some(pos) = this.buffer.find('\n') {
                let line = this.buffer[..pos].trim().to_string();
                this.buffer.drain(..=pos);

                if !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];

                if data == "[DONE]" {
                    let event = assemble_done(&mut this.tool_call_builders);
                    this.done_sent = true;
                    return Poll::Ready(Some(Ok(event)));
                }

                match serde_json::from_str::<OpenAIChunkResponse>(data) {
                    Ok(chunk) => {
                        let choice = match chunk.choices.into_iter().next() {
                            Some(c) => c,
                            None => continue,
                        };

                        if let Some(thinking) = choice.delta.reasoning
                            && !thinking.is_empty()
                        {
                            return Poll::Ready(Some(Ok(ChatStreamEvent::ThinkingDelta(thinking))));
                        }

                        if let Some(content) = choice.delta.content
                            && !content.is_empty()
                        {
                            return Poll::Ready(Some(Ok(ChatStreamEvent::ContentDelta(content))));
                        }

                        if let Some(tool_calls) = choice.delta.tool_calls {
                            for tc in tool_calls {
                                let builder = this.tool_call_builders.entry(tc.index).or_default();
                                if let Some(id) = &tc.id {
                                    builder.id = id.clone();
                                }
                                if let Some(name) = &tc.function.name {
                                    builder.name = name.clone();
                                }
                                if let Some(args) = &tc.function.arguments {
                                    builder.arguments.push_str(args);
                                    return Poll::Ready(Some(Ok(ChatStreamEvent::ToolCallArgumentDelta {
                                        id: builder.id.clone(),
                                        name: builder.name.clone(),
                                        arguments_delta: args.clone(),
                                    })));
                                }
                            }
                        }

                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse SSE chunk: {}", e);
                        continue;
                    }
                }
            }

            if this.stream_ended {
                if !this.buffer.is_empty() {
                    let remaining = std::mem::take(&mut this.buffer);
                    let line = remaining.trim();
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" && !this.done_sent {
                            let event = assemble_done(&mut this.tool_call_builders);
                            this.done_sent = true;
                            return Poll::Ready(Some(Ok(event)));
                        }
                        if let Ok(chunk) = serde_json::from_str::<OpenAIChunkResponse>(data)
                            && let Some(choice) = chunk.choices.into_iter().next()
                        {
                            if let Some(thinking) = choice.delta.reasoning
                                && !thinking.is_empty()
                            {
                                return Poll::Ready(Some(Ok(ChatStreamEvent::ThinkingDelta(
                                    thinking,
                                ))));
                            }
                            if let Some(content) = choice.delta.content
                                && !content.is_empty()
                            {
                                return Poll::Ready(Some(Ok(ChatStreamEvent::ContentDelta(
                                    content,
                                ))));
                            }
                            if let Some(tool_calls) = choice.delta.tool_calls {
                                for tc in tool_calls {
                                    let builder = this.tool_call_builders.entry(tc.index).or_default();
                                    if let Some(id) = &tc.id {
                                        builder.id = id.clone();
                                    }
                                    if let Some(name) = &tc.function.name {
                                        builder.name = name.clone();
                                    }
                                    if let Some(args) = &tc.function.arguments {
                                        builder.arguments.push_str(args);
                                        return Poll::Ready(Some(Ok(ChatStreamEvent::ToolCallArgumentDelta {
                                            id: builder.id.clone(),
                                            name: builder.name.clone(),
                                            arguments_delta: args.clone(),
                                        })));
                                    }
                                }
                            }
                        }
                    }
                }

                if !this.done_sent {
                    let event = assemble_done(&mut this.tool_call_builders);
                    this.done_sent = true;
                    return Poll::Ready(Some(Ok(event)));
                }
                return Poll::Ready(None);
            }

            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    this.buffer.push_str(&String::from_utf8_lossy(&chunk));
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(AgentError::ExecutionError(format!(
                        "Stream read error: {}",
                        e
                    )))));
                }
                Poll::Ready(None) => {
                    this.stream_ended = true;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = OpenAIClientConfig::new("key", "gpt-4");
        assert_eq!(config.api_key, "key");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn test_config_builder() {
        let config = OpenAIClientConfig::new("key", "gpt-4")
            .with_base_url("https://custom.api.com/v1")
            .with_temperature(0.7)
            .with_max_tokens(1000);

        assert_eq!(config.base_url, "https://custom.api.com/v1");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(1000));
    }

    #[test]
    fn test_client_creation() {
        let client = OpenAILLMClient::new("sk-test", "gpt-4");
        assert_eq!(client.model(), "gpt-4");
        assert_eq!(client.base_url(), "https://api.openai.com/v1");
    }

    #[test]
    fn test_client_with_config() {
        let config =
            OpenAIClientConfig::new("sk-test", "gpt-4").with_base_url("http://localhost:1234/v1");
        let client = OpenAILLMClient::with_config(config);

        assert_eq!(client.base_url(), "http://localhost:1234/v1");
    }

    #[test]
    fn test_openrouter_config() {
        let config = OpenAIClientConfig::new("sk-or-v2-xxx", "anthropic/claude-3.5-sonnet")
            .with_base_url("https://openrouter.ai/api/v1");

        let client = OpenAILLMClient::with_config(config);

        assert_eq!(client.model(), "anthropic/claude-3.5-sonnet");
        assert_eq!(client.base_url(), "https://openrouter.ai/api/v1");
    }
}
