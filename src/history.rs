use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::{ChatMessage, ToolCall};

#[derive(Debug, Clone, PartialEq)]
pub enum AgentHistoryEvent<AgentId> {
    Token {
        content: String,
    },
    TokenThinking {
        content: String,
    },
    ToolCallArgumentDelta {
        id: String,
        name: String,
        delta: String,
    },
    ToolCallStart {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolCallEnd {
        id: String,
        content: String,
        success: bool,
    },
    AgentStart {
        tool: Option<String>,
        agent: String,
        model: Option<String>,
        parent: Option<AgentId>,
    },
    AgentEnd,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistedAgentHistoryEvent<RecordId, AgentId> {
    pub id: RecordId,
    pub agent_id: AgentId,
    pub detail: AgentHistoryEvent<AgentId>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentHistoryRecord<AgentId> {
    User {
        content: String,
    },
    Assistant {
        agent_id: AgentId,
        detail: AgentHistoryEvent<AgentId>,
        model: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RehydratedHistory<AgentId> {
    pub agent_id: AgentId,
    pub history: Vec<ChatMessage>,
}

#[async_trait]
pub trait HistoryStore: Send + Sync + 'static {
    type RecordId: Send + 'static;
    type AgentId: Send + 'static;
    type Error: Send + 'static;

    async fn append_user_message(&self, content: String) -> Result<Self::RecordId, Self::Error>;

    async fn append_agent_events(
        &self,
        events: Vec<PersistedAgentHistoryEvent<Self::RecordId, Self::AgentId>>,
    ) -> Result<(), Self::Error>;

    async fn load_records(&self) -> Result<Vec<AgentHistoryRecord<Self::AgentId>>, Self::Error>;
}

pub struct HistoryPersistence<S: HistoryStore> {
    store: Arc<S>,
    recorder: Mutex<HistoryEventRecorder<S::RecordId, S::AgentId>>,
}

impl<S: HistoryStore> HistoryPersistence<S>
where
    S::AgentId: Clone + Eq + Hash + Send,
    S::RecordId: Send,
{
    pub fn new(store: S) -> Self {
        Self {
            store: Arc::new(store),
            recorder: Mutex::new(HistoryEventRecorder::new()),
        }
    }

    pub fn store(&self) -> Arc<S> {
        Arc::clone(&self.store)
    }

    pub async fn persist_user_message(&self, content: String) -> Result<S::RecordId, S::Error> {
        self.store.append_user_message(content).await
    }

    pub async fn record_event<F>(
        &self,
        agent_id: S::AgentId,
        event: AgentHistoryEvent<S::AgentId>,
        next_id: F,
    ) -> Result<(), S::Error>
    where
        S::RecordId: Send,
        S::AgentId: Clone + Eq + Hash + Send,
        F: FnMut() -> S::RecordId + Send,
    {
        let flushed = {
            let mut recorder = self.recorder.lock().await;
            recorder.push_event(agent_id, event, next_id)
        };

        if let Some(events) = flushed {
            self.store.append_agent_events(events).await?;
        }

        Ok(())
    }

    pub async fn load_rehydrated(
        &self,
        fallback_agent_id: S::AgentId,
    ) -> Result<Option<RehydratedHistory<S::AgentId>>, S::Error>
    where
        S::AgentId: Clone + Copy + Eq,
    {
        let records = self.store.load_records().await?;
        Ok(rehydrate_root_history(&records, fallback_agent_id))
    }
}

#[derive(Debug, Default)]
pub struct HistoryEventRecorder<RecordId, AgentId> {
    buffers: HashMap<AgentId, VecDeque<PersistedAgentHistoryEvent<RecordId, AgentId>>>,
    agent_models: HashMap<AgentId, String>,
}

impl<RecordId, AgentId> HistoryEventRecorder<RecordId, AgentId>
where
    AgentId: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            agent_models: HashMap::new(),
        }
    }

    pub fn push_event<F>(
        &mut self,
        agent_id: AgentId,
        detail: AgentHistoryEvent<AgentId>,
        mut next_id: F,
    ) -> Option<Vec<PersistedAgentHistoryEvent<RecordId, AgentId>>>
    where
        F: FnMut() -> RecordId,
    {
        match detail {
            AgentHistoryEvent::Token { content: delta } => {
                let buf = self.buffers.get_mut(&agent_id)?;
                let model = self.agent_models.get(&agent_id).cloned();
                match buf.back_mut() {
                    Some(PersistedAgentHistoryEvent {
                        detail: AgentHistoryEvent::Token { content },
                        ..
                    }) => content.push_str(&delta),
                    _ => buf.push_back(PersistedAgentHistoryEvent {
                        id: next_id(),
                        agent_id: agent_id.clone(),
                        detail: AgentHistoryEvent::Token { content: delta },
                        model,
                    }),
                }
                None
            }
            AgentHistoryEvent::TokenThinking { content: delta } => {
                let buf = self.buffers.get_mut(&agent_id)?;
                let model = self.agent_models.get(&agent_id).cloned();
                match buf.back_mut() {
                    Some(PersistedAgentHistoryEvent {
                        detail: AgentHistoryEvent::TokenThinking { content },
                        ..
                    }) => content.push_str(&delta),
                    _ => buf.push_back(PersistedAgentHistoryEvent {
                        id: next_id(),
                        agent_id: agent_id.clone(),
                        detail: AgentHistoryEvent::TokenThinking { content: delta },
                        model,
                    }),
                }
                None
            }
            AgentHistoryEvent::ToolCallArgumentDelta { id, name, delta } => {
                let buf = self.buffers.get_mut(&agent_id)?;
                let model = self.agent_models.get(&agent_id).cloned();
                buf.push_back(PersistedAgentHistoryEvent {
                    id: next_id(),
                    agent_id,
                    detail: AgentHistoryEvent::ToolCallArgumentDelta { id, name, delta },
                    model,
                });
                None
            }
            AgentHistoryEvent::ToolCallStart {
                id,
                name,
                arguments,
            } => self.flush_buffer(
                agent_id,
                AgentHistoryEvent::ToolCallStart {
                    id,
                    name,
                    arguments,
                },
                true,
                next_id,
            ),
            AgentHistoryEvent::ToolCallEnd {
                id,
                content,
                success,
            } => self.flush_buffer(
                agent_id,
                AgentHistoryEvent::ToolCallEnd {
                    id,
                    content,
                    success,
                },
                true,
                next_id,
            ),
            AgentHistoryEvent::AgentStart {
                tool,
                agent,
                model,
                parent,
            } => {
                if let Some(model_name) = model.clone() {
                    self.agent_models.insert(agent_id.clone(), model_name);
                }

                let buf = VecDeque::from([PersistedAgentHistoryEvent {
                    id: next_id(),
                    agent_id: agent_id.clone(),
                    detail: AgentHistoryEvent::AgentStart {
                        tool,
                        agent,
                        model: model.clone(),
                        parent,
                    },
                    model,
                }]);
                self.buffers.insert(agent_id, buf);
                None
            }
            AgentHistoryEvent::AgentEnd => {
                let flushed =
                    self.flush_buffer(agent_id.clone(), AgentHistoryEvent::AgentEnd, false, next_id);
                self.agent_models.remove(&agent_id);
                flushed
            }
        }
    }

    fn flush_buffer<F>(
        &mut self,
        agent_id: AgentId,
        detail: AgentHistoryEvent<AgentId>,
        recreate: bool,
        mut next_id: F,
    ) -> Option<Vec<PersistedAgentHistoryEvent<RecordId, AgentId>>>
    where
        F: FnMut() -> RecordId,
    {
        let mut buf = self.buffers.remove(&agent_id)?;
        let model = self.agent_models.get(&agent_id).cloned();

        if recreate {
            self.buffers.insert(agent_id.clone(), VecDeque::new());
        }

        buf.push_back(PersistedAgentHistoryEvent {
            id: next_id(),
            agent_id,
            detail,
            model,
        });

        Some(buf.into_iter().collect())
    }
}

pub fn rehydrate_root_history<AgentId>(
    records: &[AgentHistoryRecord<AgentId>],
    fallback_agent_id: AgentId,
) -> Option<RehydratedHistory<AgentId>>
where
    AgentId: Clone + Copy + Eq,
{
    let mut root_agent_id = None;
    let mut history = Vec::new();
    let mut pending = PendingAssistantMessage::default();

    for record in records {
        match record {
            AgentHistoryRecord::User { content } => {
                pending.finish_into(&mut history);
                history.push(ChatMessage::user(content.clone()));
            }
            AgentHistoryRecord::Assistant {
                agent_id, detail, ..
            } => match detail {
                AgentHistoryEvent::AgentStart {
                    tool: None,
                    parent: None,
                    ..
                } => {
                    pending.finish_into(&mut history);
                    root_agent_id.get_or_insert(*agent_id);
                }
                AgentHistoryEvent::Token { content } if Some(*agent_id) == root_agent_id => {
                    pending.content.push_str(content);
                }
                AgentHistoryEvent::ToolCallStart {
                    id,
                    name,
                    arguments,
                } if Some(*agent_id) == root_agent_id => {
                    pending.tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                }
                AgentHistoryEvent::ToolCallEnd { id, content, .. }
                    if Some(*agent_id) == root_agent_id =>
                {
                    pending.finish_into(&mut history);
                    history.push(ChatMessage::tool(id.clone(), content.clone()));
                }
                AgentHistoryEvent::AgentEnd if Some(*agent_id) == root_agent_id => {
                    pending.finish_into(&mut history);
                }
                _ => {}
            },
        }
    }

    pending.finish_into(&mut history);

    if history.is_empty() {
        return None;
    }

    Some(RehydratedHistory {
        agent_id: root_agent_id.unwrap_or(fallback_agent_id),
        history,
    })
}

#[derive(Default)]
struct PendingAssistantMessage {
    content: String,
    tool_calls: Vec<ToolCall>,
}

impl PendingAssistantMessage {
    fn finish_into(&mut self, history: &mut Vec<ChatMessage>) {
        if self.content.is_empty() && self.tool_calls.is_empty() {
            return;
        }

        history.push(ChatMessage::assistant(
            (!self.content.is_empty()).then(|| std::mem::take(&mut self.content)),
            (!self.tool_calls.is_empty()).then(|| std::mem::take(&mut self.tool_calls)),
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::json;

    use super::{
        AgentHistoryEvent, AgentHistoryRecord, HistoryEventRecorder, HistoryPersistence,
        HistoryStore, PersistedAgentHistoryEvent, rehydrate_root_history,
    };
    use crate::{ChatMessage, ToolCall};

    #[derive(Default)]
    struct InMemoryHistoryStore {
        next_user_id: Mutex<u64>,
        records: Mutex<Vec<AgentHistoryRecord<u64>>>,
        flushed: Mutex<Vec<Vec<PersistedAgentHistoryEvent<u64, u64>>>>,
    }

    #[async_trait]
    impl HistoryStore for InMemoryHistoryStore {
        type RecordId = u64;
        type AgentId = u64;
        type Error = ();

        async fn append_user_message(&self, content: String) -> Result<Self::RecordId, Self::Error> {
            let mut next_user_id = self.next_user_id.lock().expect("mutex poisoned");
            *next_user_id += 1;
            self.records
                .lock()
                .expect("mutex poisoned")
                .push(AgentHistoryRecord::User { content });
            Ok(*next_user_id)
        }

        async fn append_agent_events(
            &self,
            events: Vec<PersistedAgentHistoryEvent<Self::RecordId, Self::AgentId>>,
        ) -> Result<(), Self::Error> {
            self.flushed
                .lock()
                .expect("mutex poisoned")
                .push(events.clone());
            self.records
                .lock()
                .expect("mutex poisoned")
                .extend(events.into_iter().map(|event| AgentHistoryRecord::Assistant {
                    agent_id: event.agent_id,
                    detail: event.detail,
                    model: event.model,
                }));
            Ok(())
        }

        async fn load_records(&self) -> Result<Vec<AgentHistoryRecord<Self::AgentId>>, Self::Error> {
            Ok(self.records.lock().expect("mutex poisoned").clone())
        }
    }

    #[test]
    fn coalesces_tokens_and_flushes_on_boundaries() {
        let mut next_id = 0u64;
        let mut recorder = HistoryEventRecorder::<u64, u64>::new();

        assert_eq!(
            recorder.push_event(
                10,
                AgentHistoryEvent::AgentStart {
                    tool: None,
                    agent: "root".into(),
                    model: Some("model-a".into()),
                    parent: None,
                },
                || {
                    next_id += 1;
                    next_id
                }
            ),
            None
        );
        assert_eq!(
            recorder.push_event(
                10,
                AgentHistoryEvent::Token {
                    content: "hel".into(),
                },
                || {
                    next_id += 1;
                    next_id
                }
            ),
            None
        );
        let flushed = recorder
            .push_event(
                10,
                AgentHistoryEvent::ToolCallStart {
                    id: "call-1".into(),
                    name: "view".into(),
                    arguments: json!({"path":"README.md"}),
                },
                || {
                    next_id += 1;
                    next_id
                },
            )
            .expect("flush expected");

        assert_eq!(flushed.len(), 3);
        assert_eq!(
            flushed[1].detail,
            AgentHistoryEvent::Token {
                content: "hel".into()
            }
        );
        assert_eq!(flushed[0].model.as_deref(), Some("model-a"));
        assert_eq!(flushed[2].model.as_deref(), Some("model-a"));
    }

    #[test]
    fn rehydrates_root_history_from_persisted_events() {
        let records = vec![
            AgentHistoryRecord::User {
                content: "first".into(),
            },
            AgentHistoryRecord::Assistant {
                agent_id: 100,
                detail: AgentHistoryEvent::AgentStart {
                    tool: None,
                    agent: "panit".into(),
                    model: Some("openai/gpt-4.1-mini".into()),
                    parent: None,
                },
                model: None,
            },
            AgentHistoryRecord::Assistant {
                agent_id: 100,
                detail: AgentHistoryEvent::Token {
                    content: "hello".into(),
                },
                model: None,
            },
            AgentHistoryRecord::Assistant {
                agent_id: 100,
                detail: AgentHistoryEvent::ToolCallStart {
                    id: "call-1".into(),
                    name: "view".into(),
                    arguments: json!({"path":"README.md"}),
                },
                model: None,
            },
            AgentHistoryRecord::Assistant {
                agent_id: 200,
                detail: AgentHistoryEvent::AgentStart {
                    tool: Some("call-1".into()),
                    agent: "worker".into(),
                    model: Some("openai/gpt-4.1-mini".into()),
                    parent: Some(100),
                },
                model: None,
            },
            AgentHistoryRecord::Assistant {
                agent_id: 200,
                detail: AgentHistoryEvent::Token {
                    content: "child".into(),
                },
                model: None,
            },
            AgentHistoryRecord::Assistant {
                agent_id: 200,
                detail: AgentHistoryEvent::AgentEnd,
                model: None,
            },
            AgentHistoryRecord::Assistant {
                agent_id: 100,
                detail: AgentHistoryEvent::ToolCallEnd {
                    id: "call-1".into(),
                    content: "{\"ok\":true}".into(),
                    success: true,
                },
                model: None,
            },
            AgentHistoryRecord::Assistant {
                agent_id: 100,
                detail: AgentHistoryEvent::Token {
                    content: " world".into(),
                },
                model: None,
            },
            AgentHistoryRecord::Assistant {
                agent_id: 100,
                detail: AgentHistoryEvent::AgentEnd,
                model: None,
            },
        ];

        let rehydrated =
            rehydrate_root_history(&records, 10).expect("expected rehydrated context");

        assert_eq!(rehydrated.agent_id, 100);
        assert_eq!(
            rehydrated.history,
            vec![
                ChatMessage::user("first"),
                ChatMessage::assistant(
                    Some("hello".to_string()),
                    Some(vec![ToolCall {
                        id: "call-1".into(),
                        name: "view".into(),
                        arguments: json!({"path":"README.md"}),
                    }])
                ),
                ChatMessage::tool(Some("call-1".to_string()), "{\"ok\":true}"),
                ChatMessage::assistant(Some(" world".to_string()), None),
            ]
        );
    }

    #[tokio::test]
    async fn persistence_driver_flushes_and_rehydrates() {
        let persistence = HistoryPersistence::new(InMemoryHistoryStore::default());
        let mut next_event_id = 100u64;

        let user_id = persistence
            .persist_user_message("first".into())
            .await
            .expect("user message should persist");
        assert_eq!(user_id, 1);

        persistence
            .record_event(
                100,
                AgentHistoryEvent::AgentStart {
                    tool: None,
                    agent: "panit".into(),
                    model: Some("openai/gpt-4.1-mini".into()),
                    parent: None,
                },
                || {
                    next_event_id += 1;
                    next_event_id
                },
            )
            .await
            .expect("agent start should record");

        persistence
            .record_event(
                100,
                AgentHistoryEvent::Token {
                    content: "hello".into(),
                },
                || {
                    next_event_id += 1;
                    next_event_id
                },
            )
            .await
            .expect("token should record");

        persistence
            .record_event(
                100,
                AgentHistoryEvent::AgentEnd,
                || {
                    next_event_id += 1;
                    next_event_id
                },
            )
            .await
            .expect("agent end should flush");

        let store = persistence.store();
        let flushed = store.flushed.lock().expect("mutex poisoned");
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].len(), 3);
        drop(flushed);

        let rehydrated = persistence
            .load_rehydrated(999)
            .await
            .expect("rehydration should succeed")
            .expect("rehydration should produce history");

        assert_eq!(rehydrated.agent_id, 100);
        assert_eq!(
            rehydrated.history,
            vec![
                ChatMessage::user("first"),
                ChatMessage::assistant(Some("hello".to_string()), None),
            ]
        );
    }
}
