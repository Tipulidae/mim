from enum import Enum
from datetime import datetime

import numpy as np


class ECGStatus(Enum):
    MISSING_DATA = 0
    MISSING_LABELS = 1
    MISSING_ECG = 2
    MISSING_BEAT = 3
    MISSING_RHYTHM = 4
    MISSING_DIAGNOSES = 5
    MISSING_SUMMARY = 6
    MISSING_MEASUREMENTS = 7
    TECHNICAL_ERROR = 8
    BAD_DIAGNOSIS = 9
    FILE_MISSING = 10
    BAD_ECG_DIMENSIONS = 11
    BAD_BEAT_DIMENSIONS = 12
    EMPTY_ECG_ROWS = 13
    EMPTY_ECG_COLUMNS = 14
    EMPTY_BEAT_ROWS = 15
    EMPTY_BEAT_COLUMNS = 16
    MISSING_RECORDING = 17
    MISSING_FILE_FORMAT = 18
    MISSING_ID = 19
    MISSING_DATE = 20
    MISSING_DEVICE = 21
    MISSING_LEAD_SYSTEM = 22
    MISSING_LEAD_NAMES = 23
    BAD_LEAD_NAMES = 24
    BAD_DATE = 25
    MISSING_PATIENT = 26

    MISSING_GLASGOW = 27
    BAD_GLASGOW = 28


expected_lead_names = [
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'I',
    'II',
    'III',
    'aVR',
    'aVL',
    'aVF'
]

glasgow_vector_names = [
    'Ponset',
    'Pduration',
    'QRSonset',
    'QRSduration',
    'ST80amplitude',
    'Qduration',
    'Rduration',
    'Sduration',
    'Rprim_dur',
    'Sprim_dur',
    'Rbis_dur',
    'Sbis_dur',
    'STduration',
    'Tonset',
    'Tduration',
    'Ppos_dur',
    'Tpos_dur',
    'QRS_IntD',
    'Ppos_amp',
    'Pneg_amp',
    'pk2pkQRS_amp',
    'R1_amp',
    'Qamplitude',
    'Ramplitude',
    'Samplitude',
    'Rprim_amp',
    'Sprim_amp',
    'Rbis_amp',
    'Sbis_amp',
    'ST_amp',
    'STT28_amp',
    'STT38_amp',
    'Tpos_amp',
    'Tneg_amp',
    'QRSarea',
    'Parea',
    'Tarea',
    'Pmorphology',
    'Tmorphology',
    'Rnotch',
    'DeltaConf',
    'STslope',
    'QTinterval',
    'STM_amp',
    'ST60_amp',
    'STTmid_amp',
    'EndQRSnotch_amp',
    'PR_amp',
    'ST_ampAdj',
    'EndQRSnotchSlurOnset',
    'EndQRSnotchSlurPeak'
]

glasgow_scalar_names = [
    'QRSFrontalAxis',
    'PFrontalAxis',
    'STFrontalAxis',
    'TFrontalAxis',
    'QRSpvec48sv',
    'QRSpvec58sv',
    'QRSpvec68sv',
    'QRSpvec78sv',
    'QRSpvecMaxAmpl',
    'OverallPonset',
    'OverallPend',
    'OverallQRSonset',
    'OverallQRSend',
    'OverallTonset',
    'OverallTend',
    'HeartRateVariability',
    'StdDevNormRR',
    'LVHscore',
    'LVstrain',
    'OverallPdur',
    'OverallQRSdur',
    'OverallSTdur',
    'OverallTdur',
    'RmaxaVR',
    'RmaxaVL',
    'SampV1',
    'RampV5',
    'SampV1plusRampV5',
    'OverallPRint',
    'OverallQTint',
    'HeartRate',
    'PtermV1',
    'QTdisp',
    'QTc',
    'QTcHodge',
    'QTcBazett',
    'QTcFridericia',
    'QTcFramingham',
    'SinusRate',
    'SinusAveRR',
    'VentRate',
    'VentAveRR'
]

glasgow_diagnoses = [
    "*** ANTERIOR INFARCT - POSSIBLY ACUTE ***",
    "*** ANTEROLATERAL INFARCT - POSSIBLY ACUTE ***",
    "*** ANTEROSEPTAL INFARCT - POSSIBLY ACUTE ***",
    "*** EXTENSIVE INFARCT - POSSIBLY ACUTE ***",
    "*** INFERIOR INFARCT - POSSIBLY ACUTE ***",
    "*** LATERAL INFARCT - POSSIBLY ACUTE ***",
    "*** SEPTAL INFARCT - POSSIBLY ACUTE ***",
    "Abnormal Q waves of undetermined cause",
    "Ant/septal and lateral ST abnormality is age and gender related",
    "Ant/septal and lateral ST abnormality is nonspecific",
    "Ant/septal and lateral ST abnormality is probably due to the "
    "ventricular hypertrophy",
    "Ant/septal and lateral ST abnormality may be due to myocardial ischemia",
    "Ant/septal and lateral ST abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Ant/septal and lateral ST abnormality suggests myocardial infarct",
    "Ant/septal and lateral ST abnormality suggests myocardial "
    "injury/ischemia",
    "Ant/septal and lateral ST-T abnormality is age and gender related",
    "Ant/septal and lateral ST-T abnormality is borderline",
    "Ant/septal and lateral ST-T abnormality is nonspecific",
    "Ant/septal and lateral ST-T abnormality is probably due to the "
    "ventricular hypertrophy",
    "Ant/septal and lateral ST-T abnormality may be due to myocardial "
    "infarct or CVA",
    "Ant/septal and lateral ST-T abnormality may be due to myocardial "
    "ischemia",
    "Ant/septal and lateral ST-T abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Ant/septal and lateral ST-T abnormality suggests myocardial infarct",
    "Ant/septal and lateral ST-T abnormality suggests myocardial "
    "injury/ischemia",
    "Ant/septal and lateral T wave abnormality is age and gender related",
    "Ant/septal and lateral T wave abnormality is borderline",
    "Ant/septal and lateral T wave abnormality is nonspecific",
    "Ant/septal and lateral T wave abnormality is probably due to the "
    "ventricular hypertrophy",
    "Ant/septal and lateral T wave abnormality may be due to myocardial "
    "infarct or CVA",
    "Ant/septal and lateral T wave abnormality may be due to myocardial "
    "ischemia",
    "Ant/septal and lateral T wave abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Ant/septal and lateral T wave abnormality suggests myocardial infarct",
    "Anterior infarct - age undetermined",
    "Anterior ST abnormality is age and gender related",
    "Anterior ST abnormality is borderline",
    "Anterior ST abnormality is nonspecific",
    "Anterior ST abnormality may be due to myocardial ischemia",
    "Anterior ST abnormality may be due to the hypertrophy and/or ischemia",
    "Anterior ST abnormality suggests myocardial injury/ischemia",
    "Anterior ST elevation is nonspecific",
    "Anterior ST elevation, CONSIDER ACUTE INFARCT",
    "Anterior ST-T abnormality is age and gender related",
    "Anterior ST-T abnormality is age related : consider juvenile T waves",
    "Anterior ST-T abnormality is borderline",
    "Anterior ST-T abnormality is nonspecific",
    "Anterior ST-T abnormality may be due to myocardial ischemia",
    "Anterior ST-T abnormality may be due to the hypertrophy and/or ischemia",
    "Anterior ST-T abnormality suggests myocardial infarct",
    "Anterior ST-T abnormality suggests myocardial injury/ischemia",
    "Anterior T wave abnormality is age and gender related",
    "Anterior T wave abnormality is age related : consider juvenile T waves",
    "Anterior T wave abnormality is borderline",
    "Anterior T wave abnormality is nonspecific",
    "Anterior T wave abnormality is probably due to the ventricular "
    "hypertrophy",
    "Anterior T wave abnormality may be due to myocardial infarct or CVA",
    "Anterior T wave abnormality may be due to myocardial ischemia",
    "Anterior T wave abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Anterior T wave abnormality suggests myocardial infarct",
    "Anterior T wave abnormality suggests myocardial injury/ischemia",
    "Anterolateral infarct - age undetermined",
    "Anterolateral ST abnormality is age and gender related",
    "Anterolateral ST abnormality is nonspecific",
    "Anterolateral ST abnormality may be due to myocardial ischemia",
    "Anterolateral ST abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Anterolateral ST abnormality suggests myocardial injury/ischemia",
    "Anterolateral ST elevation - cannot rule out myocardial injury",
    "Anterolateral ST elevation is nonspecific",
    "Anterolateral ST elevation, CONSIDER ACUTE INFARCT",
    "Anterolateral ST-T abnormality is age and gender related",
    "Anterolateral ST-T abnormality is borderline",
    "Anterolateral ST-T abnormality is nonspecific",
    "Anterolateral ST-T abnormality is probably due to the ventricular "
    "hypertrophy",
    "Anterolateral ST-T abnormality may be due to myocardial infarct or CVA",
    "Anterolateral ST-T abnormality may be due to myocardial ischemia",
    "Anterolateral ST-T abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Anterolateral ST-T abnormality suggests myocardial infarct",
    "Anterolateral ST-T abnormality suggests myocardial injury/ischemia",
    "Anterolateral T wave abnormality is age and gender related",
    "Anterolateral T wave abnormality is borderline",
    "Anterolateral T wave abnormality is nonspecific",
    "Anterolateral T wave abnormality is probably due to the ventricular "
    "hypertrophy",
    "Anterolateral T wave abnormality may be due to myocardial infarct "
    "or CVA",
    "Anterolateral T wave abnormality may be due to myocardial ischemia",
    "Anterolateral T wave abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Anterolateral T wave abnormality suggests myocardial infarct",
    "Anterolateral T wave abnormality suggests myocardial injury/ischemia",
    "Anteroseptal infarct - age undetermined",
    "Anteroseptal ST abnormality is age and gender related",
    "Anteroseptal ST abnormality is borderline",
    "Anteroseptal ST abnormality is nonspecific",
    "Anteroseptal ST abnormality is probably due to the ventricular "
    "hypertrophy",
    "Anteroseptal ST abnormality may be due to myocardial ischemia",
    "Anteroseptal ST abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Anteroseptal ST abnormality suggests myocardial infarct",
    "Anteroseptal ST abnormality suggests myocardial injury/ischemia",
    "Anteroseptal ST depression is probably reciprocal to inferior infarct",
    "Anteroseptal ST elevation - cannot rule out myocardial injury",
    "Anteroseptal ST elevation is nonspecific",
    "Anteroseptal ST elevation, CONSIDER ACUTE INFARCT",
    "Anteroseptal ST-T abnormality is age and gender related",
    "Anteroseptal ST-T abnormality is borderline",
    "Anteroseptal ST-T abnormality is nonspecific",
    "Anteroseptal ST-T abnormality is probably due to the ventricular "
    "hypertrophy",
    "Anteroseptal ST-T abnormality may be due to myocardial infarct or CVA",
    "Anteroseptal ST-T abnormality may be due to myocardial ischemia",
    "Anteroseptal ST-T abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Anteroseptal ST-T abnormality suggests myocardial infarct",
    "Anteroseptal ST-T abnormality suggests myocardial injury/ischemia",
    "Anteroseptal T wave abnormality is age and gender related",
    "Anteroseptal T wave abnormality is borderline",
    "Anteroseptal T wave abnormality is nonspecific",
    "Anteroseptal T wave abnormality is probably due to the ventricular "
    "hypertrophy",
    "Anteroseptal T wave abnormality may be due to myocardial infarct or CVA",
    "Anteroseptal T wave abnormality may be due to myocardial ischemia",
    "Anteroseptal T wave abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Anteroseptal T wave abnormality suggests myocardial infarct",
    "Anteroseptal T wave abnormality suggests myocardial injury/ischemia",
    "Biventricular hypertrophy",
    "Borderline high QRS voltage - probable normal variant",
    "Borderline prolonged QT interval",
    "Cannot rule out anterior infarct - age undetermined",
    "Cannot rule out anteroseptal infarct - age undetermined",
    "Cannot rule out septal infarct - age undetermined",
    "Consider left atrial abnormality",
    "End QRS notching/slurring - early repolarization pattern",
    "Extensive infarct - age undetermined",
    "Generalized low QRS voltages",
    "Generalized low QRS voltages - consider pericardial effusion",
    "If rhythm is confirmed, the following report may not be valid",
    "Incomplete LBBB",
    "Incomplete RBBB",
    "Indeterminate axis",
    "Inferior and ant/septal ST abnormality is nonspecific",
    "Inferior and ant/septal ST abnormality may be due to myocardial "
    "ischemia",
    "Inferior and ant/septal ST abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior and ant/septal ST abnormality suggests myocardial "
    "injury/ischemia",
    "Inferior and ant/septal ST elevation - cannot rule out myocardial "
    "injury",
    "Inferior and ant/septal ST elevation is nonspecific",
    "Inferior and ant/septal ST elevation, CONSIDER ACUTE INFARCT",
    "Inferior and ant/septal ST-T abnormality is age and gender related",
    "Inferior and ant/septal ST-T abnormality is borderline",
    "Inferior and ant/septal ST-T abnormality is nonspecific",
    "Inferior and ant/septal ST-T abnormality is probably due to the "
    "ventricular hypertrophy",
    "Inferior and ant/septal ST-T abnormality may be due to myocardial "
    "infarct or CVA",
    "Inferior and ant/septal ST-T abnormality may be due to myocardial "
    "ischemia",
    "Inferior and ant/septal ST-T abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior and ant/septal ST-T abnormality suggests myocardial infarct",
    "Inferior and ant/septal ST-T abnormality suggests myocardial "
    "injury/ischemia",
    "Inferior and ant/septal T wave abnormality is age and gender related",
    "Inferior and ant/septal T wave abnormality is borderline",
    "Inferior and ant/septal T wave abnormality is nonspecific",
    "Inferior and ant/septal T wave abnormality is probably due to the "
    "ventricular hypertrophy",
    "Inferior and ant/septal T wave abnormality may be due to myocardial "
    "infarct or CVA",
    "Inferior and ant/septal T wave abnormality may be due to myocardial "
    "ischemia",
    "Inferior and ant/septal T wave abnormality may be due to the "
    "hypertrophy and/or ischemia",
    "Inferior and ant/septal T wave abnormality suggests myocardial infarct",
    "Inferior and anterior ST abnormality is nonspecific",
    "Inferior and anterior ST abnormality may be due to myocardial ischemia",
    "Inferior and anterior ST abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior and anterior ST abnormality suggests myocardial "
    "injury/ischemia",
    "Inferior and anterior ST elevation, CONSIDER ACUTE INFARCT",
    "Inferior and anterior ST-T abnormality is age and gender related",
    "Inferior and anterior ST-T abnormality is borderline",
    "Inferior and anterior ST-T abnormality is nonspecific",
    "Inferior and anterior ST-T abnormality is probably due to the "
    "ventricular hypertrophy",
    "Inferior and anterior ST-T abnormality may be due to myocardial "
    "infarct or CVA",
    "Inferior and anterior ST-T abnormality may be due to myocardial "
    "ischemia",
    "Inferior and anterior ST-T abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior and anterior ST-T abnormality suggests myocardial infarct",
    "Inferior and anterior ST-T abnormality suggests myocardial "
    "injury/ischemia",
    "Inferior and anterior T wave abnormality is age and gender related",
    "Inferior and anterior T wave abnormality is borderline",
    "Inferior and anterior T wave abnormality is nonspecific",
    "Inferior and anterior T wave abnormality is probably due to the "
    "ventricular hypertrophy",
    "Inferior and anterior T wave abnormality may be due to myocardial "
    "infarct or CVA",
    "Inferior and anterior T wave abnormality may be due to myocardial "
    "ischemia",
    "Inferior and anterior T wave abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior and anterior T wave abnormality suggests myocardial infarct",
    "Inferior and anterior T wave abnormality suggests myocardial "
    "injury/ischemia",
    "Inferior and lateral ST elevation - cannot rule out myocardial injury",
    "Inferior and lateral ST elevation is nonspecific",
    "Inferior and lateral ST elevation, CONSIDER ACUTE INFARCT",
    "Inferior and septal ST abnormality may be due to myocardial ischemia",
    "Inferior and septal ST abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior and septal ST elevation, CONSIDER ACUTE INFARCT",
    "Inferior and septal ST-T abnormality is age and gender related",
    "Inferior and septal ST-T abnormality is borderline",
    "Inferior and septal ST-T abnormality is nonspecific",
    "Inferior and septal ST-T abnormality may be due to myocardial ischemia",
    "Inferior and septal ST-T abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior and septal ST-T abnormality suggests myocardial infarct",
    "Inferior and septal ST-T abnormality suggests myocardial injury/ischemia",
    "Inferior and septal T wave abnormality is borderline",
    "Inferior and septal T wave abnormality is nonspecific",
    "Inferior and septal T wave abnormality is probably due to the "
    "ventricular hypertrophy",
    "Inferior and septal T wave abnormality may be due to myocardial "
    "ischemia",
    "Inferior and septal T wave abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior and septal T wave abnormality suggests myocardial infarct",
    "Inferior infarct - age undetermined",
    "Inferior Q waves may be due to cardiomyopathy",
    "Inferior ST abnormality is age and gender related",
    "Inferior ST abnormality is nonspecific",
    "Inferior ST abnormality is probably due to the ventricular hypertrophy",
    "Inferior ST abnormality may be age and gender related : consider "
    "normal variant",
    "Inferior ST abnormality may be due to myocardial ischemia",
    "Inferior ST abnormality may be due to the hypertrophy and/or ischemia",
    "Inferior ST abnormality suggests myocardial injury/ischemia",
    "Inferior ST elevation - cannot rule out myocardial injury",
    "Inferior ST elevation is nonspecific",
    "Inferior ST elevation, CONSIDER ACUTE INFARCT",
    "Inferior ST-T abnormality is age and gender related",
    "Inferior ST-T abnormality is borderline",
    "Inferior ST-T abnormality is nonspecific",
    "Inferior ST-T abnormality is probably due to the ventricular hypertrophy",
    "Inferior ST-T abnormality may be age and gender related : consider "
    "normal variant",
    "Inferior ST-T abnormality may be due to myocardial ischemia",
    "Inferior ST-T abnormality may be due to the hypertrophy and/or ischemia",
    "Inferior ST-T abnormality suggests myocardial infarct",
    "Inferior ST-T abnormality suggests myocardial injury/ischemia",
    "Inferior T wave abnormality is age and gender related",
    "Inferior T wave abnormality is borderline",
    "Inferior T wave abnormality is nonspecific",
    "Inferior T wave abnormality is probably due to the ventricular "
    "hypertrophy",
    "Inferior T wave abnormality may be age and gender related : consider "
    "normal variant",
    "Inferior T wave abnormality may be due to myocardial ischemia",
    "Inferior T wave abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Inferior T wave abnormality suggests myocardial infarct",
    "Inferior/lateral ST abnormality is age and gender related",
    "Inferior/lateral ST abnormality is nonspecific",
    "Inferior/lateral ST abnormality is probably due to the ventricular "
    "hypertrophy",
    "Inferior/lateral ST abnormality may be age and gender related : "
    "consider normal variant",
    "Inferior/lateral ST abnormality may be due to myocardial ischemia",
    "Inferior/lateral ST abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Inferior/lateral ST abnormality suggests myocardial injury/ischemia",
    "Inferior/lateral ST-T abnormality is age and gender related",
    "Inferior/lateral ST-T abnormality is borderline",
    "Inferior/lateral ST-T abnormality is nonspecific",
    "Inferior/lateral ST-T abnormality is probably due to the ventricular "
    "hypertrophy",
    "Inferior/lateral ST-T abnormality may be age and gender related : "
    "consider normal variant",
    "Inferior/lateral ST-T abnormality may be due to myocardial infarct "
    "or CVA",
    "Inferior/lateral ST-T abnormality may be due to myocardial ischemia",
    "Inferior/lateral ST-T abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior/lateral ST-T abnormality suggests myocardial infarct",
    "Inferior/lateral ST-T abnormality suggests myocardial injury/ischemia",
    "Inferior/lateral T wave abnormality is age and gender related",
    "Inferior/lateral T wave abnormality is borderline",
    "Inferior/lateral T wave abnormality is nonspecific",
    "Inferior/lateral T wave abnormality is probably due to the ventricular "
    "hypertrophy",
    "Inferior/lateral T wave abnormality may be age and gender related : "
    "consider normal variant",
    "Inferior/lateral T wave abnormality may be due to myocardial infarct "
    "or CVA",
    "Inferior/lateral T wave abnormality may be due to myocardial ischemia",
    "Inferior/lateral T wave abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Inferior/lateral T wave abnormality suggests myocardial infarct",
    "IV conduction defect",
    "IV conduction defect is nonspecific",
    "Lateral infarct - age undetermined",
    "Lateral Q waves may be due to cardiomyopathy",
    "Lateral ST abnormality is age and gender related",
    "Lateral ST abnormality is nonspecific",
    "Lateral ST abnormality is probably due to the ventricular hypertrophy",
    "Lateral ST abnormality may be due to myocardial ischemia",
    "Lateral ST abnormality may be due to the hypertrophy and/or ischemia",
    "Lateral ST abnormality suggests myocardial injury/ischemia",
    "Lateral ST elevation - cannot rule out myocardial injury",
    "Lateral ST elevation is nonspecific",
    "Lateral ST elevation, CONSIDER ACUTE INFARCT",
    "Lateral ST-T abnormality is age and gender related",
    "Lateral ST-T abnormality is borderline",
    "Lateral ST-T abnormality is nonspecific",
    "Lateral ST-T abnormality is probably due to the ventricular hypertrophy",
    "Lateral ST-T abnormality may be due to myocardial infarct or CVA",
    "Lateral ST-T abnormality may be due to myocardial ischemia",
    "Lateral ST-T abnormality may be due to the hypertrophy and/or ischemia",
    "Lateral ST-T abnormality suggests myocardial injury/ischemia",
    "Lateral T wave abnormality is age and gender related",
    "Lateral T wave abnormality is borderline",
    "Lateral T wave abnormality is nonspecific",
    "Lateral T wave abnormality is probably due to the ventricular "
    "hypertrophy",
    "Lateral T wave abnormality may be due to myocardial infarct or CVA",
    "Lateral T wave abnormality may be due to myocardial ischemia",
    "Lateral T wave abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Left anterior fascicular block",
    "Left axis deviation",
    "Left bundle branch block",
    "Left ventricular hypertrophy",
    "Left ventricular hypertrophy by voltage only",
    "Leftward axis",
    "Low QRS voltages in limb leads",
    "Low QRS voltages in precordial leads",
    "Marked anteroseptal ST depression accompanies the infarct",
    "Marked anteroseptal ST depression, CONSIDER ACUTE INFARCT",
    "Marked anteroseptal ST depression, CONSIDER ACUTE INFARCT (proximal "
    "LAD occlusion)",
    "Marked inferior ST depression accompanies the infarct",
    "Marked inferior ST depression, CONSIDER ACUTE INFARCT",
    "Marked lateral ST depression accompanies the infarct",
    "Marked lateral ST depression, CONSIDER ACUTE INFARCT",
    "Marked left axis deviation",
    "Marked right axis deviation",
    "Marked ST elevation - consider Brugada pattern",
    "Pacemaker rhythm - no further analysis",
    "Poor R wave progression",
    "Poor R wave progression - cannot rule out anteroseptal infarct",
    "Poor R wave progression - cannot rule out septal infarct",
    "Possible anterior infarct - age undetermined",
    "Possible anterolateral infarct - age undetermined",
    "Possible anteroseptal infarct - age undetermined",
    "Possible biatrial enlargement",
    "Possible biventricular hypertrophy",
    "Possible extensive infarct - age undetermined",
    "Possible faulty V2 - omitted from analysis",
    "Possible faulty V3 - omitted from analysis",
    "Possible faulty V4 - omitted from analysis",
    "Possible faulty V5 - omitted from analysis",
    "Possible inferior infarct - age undetermined",
    "Possible lateral infarct - age undetermined",
    "Possible left anterior fascicular block",
    "Possible left atrial abnormality",
    "Possible left posterior fascicular block",
    "Possible left ventricular hypertrophy",
    "Possible posterior extension of infarct",
    "Possible right atrial abnormality",
    "Possible right ventricular hypertrophy",
    "Possible sequence error: V1,V2 omitted",
    "Possible sequence error: V2,V3 omitted",
    "Possible sequence error: V3,V4 omitted",
    "Possible sequence error: V4,V5 omitted",
    "Prolonged QT - consider ischemia, electrolyte imbalance, drug effects",
    "Q in V1/V2 may be due to lead placement error though septal infarct "
    "cannot be excluded",
    "Q in V1/V2 may be due to LVH though septal infarct cannot be excluded",
    "Q in V1/V2 may be normal variant but septal infarct cannot be excluded",
    "QRS axis leftward for age",
    "QRS changes in V2 may be due to LVH but cannot rule out septal infarct",
    "QRS changes may be due to LVH but cannot rule out anteroseptal infarct",
    "QRS changes V3/V4 may be due to LVH but cannot rule out anterior "
    "infarct",
    "RBBB with left anterior fascicular block",
    "RBBB with RAD - possible left posterior fascicular block",
    "Right axis deviation",
    "Right bundle branch block",
    "Right ventricular hypertrophy",
    "Rightward axis",
    "rSr'(V1) - probable normal variant",
    "Septal and lateral ST abnormality is nonspecific",
    "Septal and lateral ST abnormality may be due to myocardial ischemia",
    "Septal and lateral ST abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Septal and lateral ST abnormality suggests myocardial injury/ischemia",
    "Septal and lateral ST-T abnormality is age and gender related",
    "Septal and lateral ST-T abnormality is borderline",
    "Septal and lateral ST-T abnormality is nonspecific",
    "Septal and lateral ST-T abnormality is probably due to the ventricular "
    "hypertrophy",
    "Septal and lateral ST-T abnormality may be due to myocardial ischemia",
    "Septal and lateral ST-T abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Septal and lateral ST-T abnormality suggests myocardial injury/ischemia",
    "Septal and lateral T wave abnormality is borderline",
    "Septal and lateral T wave abnormality is nonspecific",
    "Septal and lateral T wave abnormality may be due to myocardial ischemia",
    "Septal and lateral T wave abnormality may be due to the hypertrophy "
    "and/or ischemia",
    "Septal ST abnormality is age and gender related",
    "Septal ST abnormality is borderline",
    "Septal ST abnormality is nonspecific",
    "Septal ST abnormality is probably due to the ventricular hypertrophy",
    "Septal ST abnormality may be due to myocardial ischemia",
    "Septal ST abnormality may be due to the hypertrophy and/or ischemia",
    "Septal ST abnormality suggests myocardial injury/ischemia",
    "Septal ST elevation is nonspecific",
    "Septal ST elevation, CONSIDER ACUTE INFARCT",
    "Septal ST-T abnormality is age and gender related",
    "Septal ST-T abnormality is borderline",
    "Septal ST-T abnormality is nonspecific",
    "Septal ST-T abnormality is probably due to the ventricular hypertrophy",
    "Septal ST-T abnormality may be due to myocardial ischemia",
    "Septal ST-T abnormality may be due to the hypertrophy and/or ischemia",
    "Septal ST-T abnormality suggests myocardial injury/ischemia",
    "Septal T wave abnormality is age and gender related",
    "Septal T wave abnormality is borderline",
    "Septal T wave abnormality is nonspecific",
    "Septal T wave abnormality is probably due to the ventricular hypertrophy",
    "Septal T wave abnormality may be due to myocardial ischemia",
    "Septal T wave abnormality may be due to the hypertrophy and/or ischemia",
    "Severe right axis deviation",
    "Short PR interval",
    "Short QT interval",
    "Small inferior Q waves noted: probably normal ECG",
    "Small inferior Q waves: infarct cannot be excluded",
    "Small lateral Q waves noted: probably normal ECG",
    "ST junctional depression is borderline",
    "ST junctional depression is nonspecific",
    "Tall R V1/V2 probably reflect the infarct",
    "Tall T waves - consider acute ischemia or hyperkalemia",
    "Tall T waves - consider hyperkalemia",
    "V1/V2 are at least one interspace too high and have been omitted from "
    "the analysis",
    "Widespread ST abnormality is nonspecific",
    "Widespread ST abnormality may be due to myocardial ischemia",
    "Widespread ST abnormality may be due to the hypertrophy and/or ischemia",
    "Widespread ST abnormality suggests myocardial injury/ischemia",
    "Widespread ST depression, CONSIDER ACUTE INFARCT (left main occlusion "
    "/ multivessel disease)",
    "Widespread ST elevation - consider pericarditis",
    "Widespread ST elevation is nonspecific",
    "Widespread ST elevation suggests pericarditis",
    "Widespread ST elevation, CONSIDER ACUTE INFARCT",
    "Widespread ST-T abnormality is age and gender related",
    "Widespread ST-T abnormality is borderline",
    "Widespread ST-T abnormality is nonspecific",
    "Widespread ST-T abnormality is probably due to the ventricular "
    "hypertrophy",
    "Widespread ST-T abnormality may be due to myocardial infarct or CVA",
    "Widespread ST-T abnormality may be due to myocardial ischemia",
    "Widespread ST-T abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Widespread ST-T abnormality suggests myocardial infarct",
    "Widespread ST-T abnormality suggests myocardial injury/ischemia",
    "Widespread T wave abnormality is age and gender related",
    "Widespread T wave abnormality is borderline",
    "Widespread T wave abnormality is nonspecific",
    "Widespread T wave abnormality is probably due to the ventricular "
    "hypertrophy",
    "Widespread T wave abnormality may be due to myocardial infarct or CVA",
    "Widespread T wave abnormality may be due to myocardial ischemia",
    "Widespread T wave abnormality may be due to the hypertrophy and/or "
    "ischemia",
    "Widespread T wave abnormality suggests myocardial infarct",
    "WPW pattern - probable anteroseptal accessory pathway",
    "WPW pattern - probable left anterolateral accessory pathway",
    "WPW pattern - probable left posterolateral accessory pathway",
    "WPW pattern - probable left posteroseptal accessory pathway",
    "WPW pattern - probable midseptal accessory pathway",
    "WPW pattern - probable right anterolateral accessory pathway",
    "WPW pattern - probable right posterolateral accessory pathway",
    "WPW pattern - probable right posteroseptal accessory pathway",
]
glasgow_diagnoses_index = {
    diagnosis: index for index, diagnosis in enumerate(glasgow_diagnoses)
}


def ecg_status(ecg):
    return (
        data_status(ecg) |
        measurement_status(ecg) |
        patient_status(ecg) |
        recording_status(ecg)
    )


def data_status(ecg):
    if 'Data' not in ecg:
        return {ECGStatus.MISSING_DATA}

    data = ecg['Data']
    return (
        label_status(data) |
        raw_status(data) |
        beat_status(data) |
        lead_name_status(data)
    )


def measurement_status(ecg):
    if 'Measurements' not in ecg:
        return {ECGStatus.MISSING_MEASUREMENTS}

    measurements = ecg['Measurements']
    return (
        diagnose_status(measurements) |
        rhythm_status(measurements) |
        summary_status(measurements) |
        glasgow_status(measurements)
    )


def patient_status(ecg):
    if 'Patient' not in ecg:
        return {ECGStatus.MISSING_PATIENT}

    patient = ecg['Patient']

    alias = flatten_nested(extract_field(patient, 'ID'))
    if len(alias) == 0:
        return {ECGStatus.MISSING_ID}

    return set()


def recording_status(ecg):
    if 'Recording' not in ecg:
        return {ECGStatus.MISSING_RECORDING}

    recording = ecg['Recording']
    return (
        date_status(recording) |
        device_status(recording) |
        lead_system_status(recording)
    )


def label_status(data):
    if 'Labels' not in data.dtype.fields:
        return {ECGStatus.MISSING_LABELS}

    return set()


def raw_status(data):
    if 'ECG' not in data.dtype.fields:
        return {ECGStatus.MISSING_ECG}

    ecg = extract_field(data, 'ECG')[0][0]
    if is_ecg_shape_malformed(ecg.shape):
        return {ECGStatus.BAD_ECG_DIMENSIONS}

    status = set()
    ecg = ecg[:10000, :8]
    if empty_row_count(ecg) > 0:
        status.add(ECGStatus.EMPTY_ECG_ROWS)
    if empty_row_count(ecg.T) > 0:
        status.add(ECGStatus.EMPTY_ECG_COLUMNS)

    return status


def beat_status(data):
    if 'Beat' not in data.dtype.fields:
        return {ECGStatus.MISSING_BEAT}

    beat = extract_field(data, 'Beat')[0][0]
    if is_beat_shape_malformed(beat.shape):
        return {ECGStatus.BAD_BEAT_DIMENSIONS}

    status = set()
    beat = beat[:1200, :8]
    if empty_row_count(beat) > 0:
        status.add(ECGStatus.EMPTY_BEAT_ROWS)
    if empty_row_count(beat.T) > 0:
        status.add(ECGStatus.EMPTY_BEAT_COLUMNS)

    return status


def diagnose_status(measurements):
    diagnoses = flatten_nested(extract_field(measurements, 'D'))
    if len(diagnoses) == 0:
        return {ECGStatus.MISSING_DIAGNOSES}

    if any(is_bad_diagnose(d) for d in diagnoses):
        return {ECGStatus.BAD_DIAGNOSIS}

    return set()


def rhythm_status(measurements):
    rhythms = flatten_nested(extract_field(measurements, 'R'))
    if len(rhythms) == 0:
        return {ECGStatus.MISSING_RHYTHM}

    return set()


def summary_status(measurements):
    summary = flatten_nested(extract_field(measurements, 'S'))
    if len(summary) == 0:
        return {ECGStatus.MISSING_SUMMARY}

    if 'Technical error' in summary:
        return {ECGStatus.TECHNICAL_ERROR}

    return set()


def glasgow_status(measurements):
    try:
        glasgow_data = measurements[0][0]
    except IndexError:
        return {ECGStatus.MISSING_GLASGOW}

    return (
        glasgow_scalar_status(glasgow_data) |
        glasgow_vector_status(glasgow_data)
    )


def glasgow_scalar_status(glasgow_data):
    all_names = glasgow_data.dtype.names
    for name in glasgow_scalar_names:
        if name not in all_names:
            return {ECGStatus.MISSING_GLASGOW}

    for name in glasgow_scalar_names:
        if len(glasgow_data[name][0]) != 1:
            return {ECGStatus.BAD_GLASGOW}

    return set()


def glasgow_vector_status(glasgow_data):
    all_names = glasgow_data.dtype.names
    for name in glasgow_vector_names:
        if name not in all_names:
            return {ECGStatus.MISSING_GLASGOW}

    for name in glasgow_vector_names:
        if len(glasgow_data[name][0]) != 12:
            return {ECGStatus.BAD_GLASGOW}

    return set()


def date_status(recording):
    try:
        date = extract_field(recording, 'Date')[0][0][0]
    except IndexError:
        return {ECGStatus.MISSING_DATE}

    try:
        datetime.strptime(date, "%d-%b-%Y %H:%M:%S")
    except ValueError:
        return {ECGStatus.BAD_DATE}

    return set()


def device_status(recording):
    try:
        extract_field(recording, 'Device')[0][0][0]
    except IndexError:
        return {ECGStatus.MISSING_DEVICE}

    return set()


def lead_system_status(recording):
    try:
        extract_field(recording, 'Lead_system')[0][0][0]
    except IndexError:
        return {ECGStatus.MISSING_LEAD_SYSTEM}

    return set()


def lead_name_status(data):
    try:
        labels = extract_field(data, 'Labels')[0][0][0]
        actual_lead_names = [x[0] for x in labels]
    except IndexError:
        return {ECGStatus.MISSING_LEAD_NAMES}

    if actual_lead_names != expected_lead_names:
        return {ECGStatus.BAD_LEAD_NAMES}

    return set()


def extract_field(data, field):
    # Given some data, check that it's a numpy-array with a field
    # with the given name, then return the contents of that field, otherwise
    # return an empty list.
    if not isinstance(data, np.ndarray):
        return []

    dtypes = data.dtype.fields
    if dtypes is None:
        return []

    if field not in dtypes:
        return []

    return data[field]


def is_empty_nested(data):
    return len(data[0][0]) == 0


def flatten_nested(data):
    if len(data) == 0:
        return []
    if is_empty_nested(data):
        return []
    else:
        return [x[0] for x in data[0][0][0]]


def contains_bad_diagnosis(diagnoses):
    for diagnose in diagnoses:
        if is_bad_diagnose(diagnose):
            return True
    return False


def is_bad_diagnose(diagnose):
    return (
        diagnose.startswith("Lead(s) unsuitable for analysis:") or
        diagnose in bad_diagnoses
    )


bad_diagnoses = {
    '--- No further analysis made ---',
    '--- Possible arm lead reversal - hence only aVF, V1-V6 analyzed ---',
    '--- Possible arm/leg lead interchange - hence only V1-V6 analyzed ---',
    '--- Possible limb lead reversal - hence only V1-V6 analyzed ---',
    '--- Possible measurement error ---',
    '--- Similar QRS in V leads ---',
    '--- Suggests dextrocardia ---',
    '--- Technically unsatisfactory tracing ---'
    'IV conduction defect',
    'Possible faulty V2 - omitted from analysis',
    'Possible faulty V3 - omitted from analysis',
    'Possible faulty V4 - omitted from analysis',
    'Possible faulty V5 - omitted from analysis',
    'Possible sequence error: V1,V2 omitted',
    'Possible sequence error: V2,V3 omitted',
    'Possible sequence error: V3,V4 omitted',
    'Possible sequence error: V4,V5 omitted',
    'V1/V2 are at least one interspace too high and have been omitted from '
    'the analysis'
}


def is_ecg_shape_malformed(shape):
    return (
        len(shape) != 2 or
        shape[0] < 10000 or
        shape[1] < 8
    )


def is_beat_shape_malformed(shape):
    return (
        len(shape) != 2 or
        shape[0] < 1200 or
        shape[1] < 8
    )


def empty_row_count(x):
    return len(x) - x.any(axis=1).sum()
