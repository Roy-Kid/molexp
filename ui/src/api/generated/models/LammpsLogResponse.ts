/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { LammpsThermoStage } from './LammpsThermoStage';
/**
 * Parsed LAMMPS log thermo stages, produced by ``molpy.io.LAMMPSLog``.
 */
export type LammpsLogResponse = {
    nStages?: number;
    path: string;
    stages?: Array<LammpsThermoStage>;
    version?: (string | null);
};

