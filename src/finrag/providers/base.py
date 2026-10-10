"""Chat factory contract at the LangChain infrastructure boundary."""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from finrag.config import ChatModelSettings

# Preserve Runnable composition, message responses, and with_structured_output.
# Concrete models must implement the structured-output support their callers need;
# inheriting BaseChatModel alone does not guarantee that support.
type ChatProviderFactory = Callable[[ChatModelSettings], BaseChatModel]
