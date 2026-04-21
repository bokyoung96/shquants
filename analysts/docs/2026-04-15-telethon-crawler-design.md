# Telethon DOC_POOL Crawler Design

## Goal
Add a Telethon-based Telegram channel crawler inside `analysts/` that follows new `DOC_POOL` posts going forward, downloads PDF documents only, persists crawl state safely, and feeds the existing ARAS pipeline.

## Scope
- Replace the current Bot API polling assumption for the real crawl path with a Telethon user-session client.
- Keep all work inside `analysts/`.
- Crawl only **new posts going forward**.
- Accept **PDF documents only**.
- Preserve existing parser/router/agent/wiki/signal pipeline behavior.

## Constraints
- Secrets and session artifacts must stay out of git.
- First-run authentication requires a Telegram login code and optional 2FA password.
- Offset-style Bot API update tracking does not fit Telethon channel crawling; state must be message-based.
- Existing deterministic/idempotent behavior must remain true.

## Recommended architecture
1. Add a local config surface (`analysts/config.local.json`) for Telethon runtime settings.
2. Add a Telethon-specific client/adapter layer that resolves the target channel and iterates new messages.
3. Track per-channel crawl state via `last_seen_message_id` in SQLite state.
4. Convert qualifying Telegram messages into the existing `ReportRecord` contract.
5. Run the existing parse → route → analyze → wiki → signal steps unchanged after download persistence.

## Data/state contract
### Local config
A gitignored config file will hold:
- `telegram.api_id`
- `telegram.api_hash`
- `telegram.phone_number`
- `telegram.session_name`
- `telegram.channel`
- `telegram.mode` (`telethon`)
- `telegram.pdf_only` (`true`)

### Crawl state
Replace/update state tracking to support:
- `telethon.last_seen_message_id.<channel>`

Semantics:
- On first successful channel inspection in `new_posts_only` mode, seed the last-seen id to the channel's current latest message id without downloading older posts.
- On later runs, process only messages with ids greater than the stored value.
- Advance state only after the message is downloaded and recorded successfully.

## Auth flow
- `auth-login` CLI command opens a Telethon login flow.
- User enters the Telegram code interactively at the terminal.
- If Telegram account 2FA is enabled, prompt for the password.
- Telethon stores a local session file under `analysts/data/state/`.

## Failure handling
- Invalid/missing local config: fail clearly before network work begins.
- Auth failure: preserve no partial crawl state updates.
- Channel resolution failure: fail clearly and keep prior state.
- Download/persist failure: do not advance `last_seen_message_id` for that message.
- Non-PDF posts: ignore and do not treat as downloads.

## Testing strategy
- Config loader tests for local config + defaults.
- Crawl-state tests for first-run seeding and subsequent new-post filtering.
- Fetcher tests for PDF filtering and no-advance-on-failure behavior.
- CLI tests for config inspection and auth command plumbing where feasible.
- Existing end-to-end pipeline tests remain green.

## Delivery split
- Lane 1: config, secrets hygiene, Telethon client/auth plumbing
- Lane 2: Telethon fetcher, crawl state semantics, persistence
- Lane 3: CLI integration, docs, tests, verification
