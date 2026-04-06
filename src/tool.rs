use std::{
    borrow::Cow,
    collections::{HashMap, hash_map::Values},
    sync::Arc,
};

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

use crate::AgentContext;

/// Type-erased tool stored in the registry.
///
/// Accepts both concrete tool types and pre-boxed `Arc<dyn Tool<C> + Send + Sync>`
/// via `From`/`Into` conversions.
pub struct DynTool<C>(pub Arc<dyn Tool<C> + Send + Sync>);

impl<C> Clone for DynTool<C> {
    fn clone(&self) -> Self {
        DynTool(self.0.clone())
    }
}

impl<C> std::ops::Deref for DynTool<C> {
    type Target = dyn Tool<C> + Send + Sync;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<C> DynTool<C> {
    pub fn new<T: Tool<C> + Send + Sync + 'static>(tool: T) -> Self {
        DynTool(Arc::new(tool))
    }
}

impl<C> From<Arc<dyn Tool<C> + Send + Sync>> for DynTool<C> {
    fn from(arc: Arc<dyn Tool<C> + Send + Sync>) -> Self {
        DynTool(arc)
    }
}

#[async_trait]
pub trait Tool<C> {
    fn id(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn execute(&self, context: &ToolContext<C>) -> ToolResult;
}

/// Context of tool execution
pub struct ToolContext<'a, C> {
    /// ID of tool call
    pub id: String,
    pub args: Value,
    /// Context of agent this tool call was made on
    pub agent: &'a C,
}

#[derive(Debug, Clone, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArguments(Cow<'static, str>),
    #[error("execution failed: {0}")]
    ExecutionFailed(Cow<'static, str>),
    #[error("not found: {0}")]
    NotFound(Cow<'static, str>),
    #[error("permission denied: {0}")]
    PermissionDenied(Cow<'static, str>),
}

impl ToolError {
    pub fn invalid_arguments(message: impl Into<Cow<'static, str>>) -> ToolError {
        Self::InvalidArguments(message.into())
    }

    pub fn execution_failed(message: impl Into<Cow<'static, str>>) -> ToolError {
        Self::ExecutionFailed(message.into())
    }

    pub fn not_found(message: impl Into<Cow<'static, str>>) -> ToolError {
        Self::NotFound(message.into())
    }

    pub fn permission_denied(message: impl Into<Cow<'static, str>>) -> ToolError {
        Self::PermissionDenied(message.into())
    }
}

pub type ToolResult = Result<String, ToolError>;

#[derive(Clone)]
pub struct ToolRegistry<C> {
    tools: Arc<HashMap<String, DynTool<C>>>,
}

impl<C> ToolRegistry<C> {
    pub fn new() -> ToolRegistry<C> {
        Self {
            tools: Arc::new(HashMap::new()),
        }
    }

    pub fn builder() -> ToolRegistryBuilder<C> {
        ToolRegistryBuilder::new()
    }

    pub fn get(&self, id: &str) -> Option<&DynTool<C>> {
        self.tools.get(id)
    }

    pub async fn execute<'a>(
        &'a self,
        id: &str,
        context: &'a ToolContext<'a, C>,
    ) -> Option<ToolResult> {
        Some(self.tools.get(id)?.execute(context).await)
    }

    pub fn all<'a>(&'a self) -> Values<'a, String, DynTool<C>> {
        self.tools.values()
    }
}

impl<C> Default for ToolRegistry<C> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ToolRegistryBuilder<C> {
    tools: HashMap<String, DynTool<C>>,
}

impl<C> ToolRegistryBuilder<C> {
    pub fn new() -> ToolRegistryBuilder<C> {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(mut self, tool: impl Into<DynTool<C>>) -> ToolRegistryBuilder<C> {
        let tool = tool.into();
        self.tools.insert(tool.id().to_string(), tool);
        self
    }

    pub fn build(self) -> ToolRegistry<C> {
        ToolRegistry {
            tools: Arc::new(self.tools),
        }
    }
}

impl<C> Default for ToolRegistryBuilder<C> {
    fn default() -> Self {
        Self::new()
    }
}
