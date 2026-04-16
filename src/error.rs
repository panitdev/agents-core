use thiserror::Error;

/// Result type returned by LLM operations.
pub type LLMResult<T> = Result<T, LLMError>;

/// Errors that can occur during LLM operations.
///
/// # Variants
///
/// - `RequestFailed`: HTTP request failed (network error, timeout, etc.)
/// - `ApiError`: API returned an error response (non-2xx status code)
/// - `ParseError`: Failed to parse response body as JSON
/// - `NoChoices`: API response contained no choices (empty response)
/// - `StreamError`: Error while reading response stream
///
/// # Examples
///
/// ```
/// use panit_agents_core::LLMError;
///
/// let err = LLMError::NoChoices;
/// assert!(err.to_string().contains("no choices"));
/// ```
#[derive(Error, Debug, Clone)]
pub enum LLMError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(String),

    #[error("API error {status}: {message}")]
    ApiError { status: u16, message: String },

    #[error("parse error: {0}")]
    ParseError(String),

    #[error("no choices in response")]
    NoChoices,

    #[error("stream error: {0}")]
    StreamError(String),
}
