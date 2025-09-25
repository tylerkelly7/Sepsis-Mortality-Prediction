SELECT 
    ad.subject_id,
    icu.stay_id,
    icu.hospital_expire_flag,
    icu.admittime,
    icu.dischtime,
    MAX(age.age) AS max_age,

    icu.los_icu,
    icu.first_icu_stay,

    soi.suspected_infection,
    sepsis.sofa_score,
    sepsis.sepsis3,

    AVG(uo.urineoutput) AS avg_urineoutput,

    MIN(vital.glucose) AS glucose_min,
    MAX(vital.glucose) AS glucose_max,
    AVG(vital.glucose) AS glucose_average,

    MIN(chemistry.sodium) AS sodium_min,
    MAX(chemistry.sodium) AS sodium_max,
    AVG(chemistry.sodium) AS sodium_average,

    charlson.diabetes_without_cc,
    charlson.diabetes_with_cc,
    charlson.severe_liver_disease,
    charlson.aids,
    charlson.renal_disease,

    MIN(vital.heart_rate) AS heart_rate_min,
    MAX(vital.heart_rate) AS heart_rate_max,
    AVG(vital.heart_rate) AS heart_rate_mean,

    MIN(vital.sbp) AS sbp_min,
    MAX(vital.sbp) AS sbp_max,
    AVG(vital.sbp) AS sbp_mean,

    MIN(vital.dbp) AS dbp_min,
    MAX(vital.dbp) AS dbp_max,
    AVG(vital.dbp) AS dbp_mean,

    MIN(vital.resp_rate) AS resp_rate_min,
    MAX(vital.resp_rate) AS resp_rate_max,
    AVG(vital.resp_rate) AS resp_rate_mean,

    MIN(vital.spo2) AS spo2_min,
    MAX(vital.spo2) AS spo2_max,
    AVG(vital.spo2) AS spo2_mean,

    MAX(gcs.gcs) AS coma_score,
    AVG(chemistry.albumin) as avg_albumin,
    ad.race,
    soi.antibiotic,
    pa.gender

FROM `physionet-data.mimiciv_hosp.admissions` AS ad

JOIN `physionet-data.mimiciv_hosp.patients` AS pa
    ON ad.subject_id = pa.subject_id

JOIN `physionet-data.mimiciv_derived.age` AS age
    ON ad.subject_id = age.subject_id

JOIN `physionet-data.mimiciv_derived.icustay_detail` AS icu
    ON ad.hadm_id = icu.hadm_id

JOIN `physionet-data.mimiciv_derived.chemistry` AS chemistry
    ON ad.hadm_id = chemistry.hadm_id

JOIN `physionet-data.mimiciv_derived.charlson` AS charlson
    ON ad.hadm_id = charlson.hadm_id

JOIN `physionet-data.mimiciv_derived.gcs` AS gcs
    ON icu.stay_id = gcs.stay_id

JOIN `physionet-data.mimiciv_derived.suspicion_of_infection` AS soi
    ON ad.hadm_id = soi.hadm_id

JOIN `physionet-data.mimiciv_derived.sepsis3` AS sepsis
    ON icu.stay_id = sepsis.stay_id

JOIN `physionet-data.mimiciv_derived.sofa` AS sofa
    ON icu.stay_id = sofa.stay_id

JOIN `physionet-data.mimiciv_derived.urine_output` AS uo
    ON icu.stay_id = uo.stay_id

JOIN `physionet-data.mimiciv_derived.vitalsign` AS vital
    ON icu.stay_id = vital.stay_id

WHERE 
    age.age >= 18
    AND icu.los_icu >= 1
    AND sepsis.sepsis3 = TRUE
    AND sepsis.sofa_score >= 2
    AND soi.suspected_infection = 1
    AND sofa.pao2fio2ratio_novent <= 200

GROUP BY ad.subject_id, icu.stay_id, icu.hospital_expire_flag, icu.admittime, icu.dischtime, icu.los_icu, icu.first_icu_stay,
         soi.suspected_infection, sepsis.sofa_score, sepsis.sepsis3, charlson.diabetes_without_cc, charlson.diabetes_with_cc,
         charlson.severe_liver_disease, charlson.aids, charlson.renal_disease, gcs.gcs, ad.race, soi.antibiotic, pa.gender
