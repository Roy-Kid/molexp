/**
 * Context resolver
 * 
 * Resolves Context from:
 * - run.json files (context field)
 * - checkpoint files
 */

import { BaseResolver } from './BaseResolver';
import type { Context } from '../types/documents';
import contextSchema from '@schemas/context.json';

export class ContextResolver extends BaseResolver<Context> {
    constructor() {
        super(contextSchema);
    }
}

export const contextResolver = new ContextResolver();
