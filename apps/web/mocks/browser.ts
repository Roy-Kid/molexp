/**
 * MSW browser worker setup for development
 */

import { setupWorker } from "msw/browser";
import { handlers } from "./handlers";

export const worker = setupWorker(...handlers);

/**
 * Start the MSW worker
 */
export async function start() {
    return worker.start({
        onUnhandledRequest: "bypass",
    });
}
