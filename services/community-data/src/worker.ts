import { GC_AFTER_SECONDS } from "./caps";
import { handleFetch } from "./routes";
import { gcStale, rebuildCaches } from "./store";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handleFetch(request, env);
  },

  // Hourly: rebuild cached index/dataset responses, then drop
  // incomplete submissions older than the documented GC window.
  async scheduled(
    _controller: ScheduledController,
    env: Env,
    _ctx: ExecutionContext,
  ): Promise<void> {
    const now = Math.floor(Date.now() / 1000);
    await rebuildCaches(env.DB, now);
    await gcStale(env.DB, now - GC_AFTER_SECONDS);
  },
};
