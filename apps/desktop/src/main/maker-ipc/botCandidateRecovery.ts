import { isNetworkishErrorMessage, isOverloadErrorMessage } from '@cindy/maker-core';
import { matchesDeterministicUsageExhaustionText } from '@cindy/maker-shared/error-redaction';
import type { InterruptedTurnErrorSignals } from './interruptedTurnAutoResume.js';

/**
 * Account/budget exhaustion: every candidate on the same account fails the same way,
 * so only a candidate on another account may take over (#5266). Structured tags are
 * authoritative; the shared deterministic text is the fallback for untagged envelopes.
 */
export function isAccountUsageExhausted(signals: InterruptedTurnErrorSignals): boolean {
  return signals.codexErrorInfo === 'usageLimitExceeded' || signals.sdkError === 'billing_error'
    || matchesDeterministicUsageExhaustionText(signals.message ?? '');
}

type AccountRoute = { harness: string; providerId?: string | null; agentDeviceId?: string | null };

/** Same harness, source and device: an exhausted account cannot be recovered by this route. */
export function isSameAccountRoute(candidate: AccountRoute, current: AccountRoute): boolean {
  return candidate.harness === current.harness
    && (candidate.providerId ?? null) === (current.providerId ?? null)
    && (candidate.agentDeviceId ?? null) === (current.agentDeviceId ?? null);
}

/** A failed provider may be replaced, even when retrying that provider is forbidden. */
export function isBotCandidateUnavailable(signals: InterruptedTurnErrorSignals): boolean {
  const reason = signals.reason ?? '';
  // Explicit product/control failures must never be reinterpreted from their text.
  if (reason && ![
    'pi-gateway-drop', 'upstream-overload', 'empty-response', 'turn-failed',
    'provider_auth_or_access', 'provider_quota_limit', 'provider_rate_limit',
    'provider_server_error', 'model_unavailable', 'user_model_access_denied', 'agent-start-failed',
  ].includes(reason)) return false;
  if (reason && reason !== 'turn-failed') return true;
  if (isAccountUsageExhausted(signals)) return true;
  const tag = signals.sdkError ?? '';
  if (['authentication_failed', 'authentication_error', 'rate_limit', 'server_error', 'billing_error',
    'model_not_found', 'user_model_access_denied', 'agent_start_failed'].includes(tag)) return true;
  const message = signals.message ?? '';
  if (/user (?:denied|rejected)|approval (?:denied|required)|context.{0,20}(?:overflow|too long)|prompt too long/i.test(message)) return false;
  if ([401, 402, 403, 429, 500, 502, 503, 504, 529].includes(signals.errorStatus ?? 0)) return true;
  if (/permission denied/i.test(message)) return false;
  // Some Claude session-limit envelopes omit both HTTP status and SDK classification.
  // A known input/control error must not be reclassified from this text alone.
  const sessionLimit = (!tag || tag === 'unknown') && /\byou['’]ve hit your session limit\b/i.test(message);
  return sessionLimit || isNetworkishErrorMessage(message) || isOverloadErrorMessage(message, signals.errorStatus)
    || /invalid api key|missing bearer token|no available oauth accounts|credit balance too low|rate.limit exceeded|model.{0,80}(?:not found|does not exist|unavailable)|(?:failed to (?:start|spawn)|could not start) (?:the )?(?:agent|pi|codex|claude)/i.test(message);
}

/** The Session reference is only a teardown lease, never proof that a route changed. */
export interface RuntimeFallbackResult<T> {
  session: T | null;
  outcome: 'unchanged' | 'switched' | 'exhausted' | 'failed' | 'superseded';
}

export function canResumeAfterRuntimeFallback(
  requireRouteChange: boolean,
  result: RuntimeFallbackResult<unknown>,
): boolean {
  return result.outcome !== 'superseded' && (!requireRouteChange || result.outcome === 'switched');
}
