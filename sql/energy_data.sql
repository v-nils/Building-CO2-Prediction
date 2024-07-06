-- Step 1: Add temporary columns
ALTER TABLE energy_data
ADD COLUMN temp_hospital_general_medical_surgical_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_financial_office_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_retail_store_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_bank_branch_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_office_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_other_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_parking_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_enclosed_mall_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_energy_power_station_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_fitness_center_health_club_gym_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_food_sales_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_food_service_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_medical_office_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_urgent_care_clinic_other_outpatient_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_hotel_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_k_12_school_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_pre_school_daycare_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_laboratory_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_mailing_center_post_office_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_movie_theater_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_manufacturing_industrial_plant_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_multifamily_housing_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_residence_hall_dormitory_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_non_refrigerated_warehouse_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_museum_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_performing_arts_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_refrigerated_warehouse_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_restaurant_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_self_storage_facility_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_senior_living_community_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_social_meeting_hall_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_supermarket_grocery_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_wholesale_club_supercenter_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_wastewater_treatment_plant_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_worship_facility_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_adult_education_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_ambulatory_surgical_center_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_automobile_dealership_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_barracks_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_college_university_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_convention_center_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_courthouse_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_data_center_gross_floor_area_ft DOUBLE PRECISION,
ADD COLUMN temp_distribution_center_gross_floor_area_ft DOUBLE PRECISION;

-- Step 2: Update temporary columns with casted values, setting invalid values to NULL
UPDATE your_table_name
SET
    temp_hospital_general_medical_surgical_gross_floor_area_ft = CASE
        WHEN hospital_general_medical_surgical_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN hospital_general_medical_surgical_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_financial_office_gross_floor_area_ft = CASE
        WHEN financial_office_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN financial_office_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_retail_store_gross_floor_area_ft = CASE
        WHEN retail_store_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN retail_store_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_bank_branch_gross_floor_area_ft = CASE
        WHEN bank_branch_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN bank_branch_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_office_gross_floor_area_ft = CASE
        WHEN office_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN office_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_other_gross_floor_area_ft = CASE
        WHEN other_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN other_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_parking_gross_floor_area_ft = CASE
        WHEN parking_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN parking_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_enclosed_mall_gross_floor_area_ft = CASE
        WHEN enclosed_mall_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN enclosed_mall_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_energy_power_station_gross_floor_area_ft = CASE
        WHEN energy_power_station_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN energy_power_station_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_fitness_center_health_club_gym_gross_floor_area_ft = CASE
        WHEN fitness_center_health_club_gym_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN fitness_center_health_club_gym_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_food_sales_gross_floor_area_ft = CASE
        WHEN food_sales_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN food_sales_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_food_service_gross_floor_area_ft = CASE
        WHEN food_service_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN food_service_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_medical_office_gross_floor_area_ft = CASE
        WHEN medical_office_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN medical_office_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_urgent_care_clinic_other_outpatient_gross_floor_area_ft = CASE
        WHEN urgent_care_clinic_other_outpatient_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN urgent_care_clinic_other_outpatient_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_hotel_gross_floor_area_ft = CASE
        WHEN hotel_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN hotel_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_k_12_school_gross_floor_area_ft = CASE
        WHEN k_12_school_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN k_12_school_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_pre_school_daycare_gross_floor_area_ft = CASE
        WHEN pre_school_daycare_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN pre_school_daycare_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_laboratory_gross_floor_area_ft = CASE
        WHEN laboratory_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN laboratory_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_mailing_center_post_office_gross_floor_area_ft = CASE
        WHEN mailing_center_post_office_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN mailing_center_post_office_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_movie_theater_gross_floor_area_ft = CASE
        WHEN movie_theater_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN movie_theater_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_manufacturing_industrial_plant_gross_floor_area_ft = CASE
        WHEN manufacturing_industrial_plant_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN manufacturing_industrial_plant_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_multifamily_housing_gross_floor_area_ft = CASE
        WHEN multifamily_housing_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN multifamily_housing_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_residence_hall_dormitory_gross_floor_area_ft = CASE
        WHEN residence_hall_dormitory_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN residence_hall_dormitory_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_non_refrigerated_warehouse_gross_floor_area_ft = CASE
        WHEN non_refrigerated_warehouse_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN non_refrigerated_warehouse_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_museum_gross_floor_area_ft = CASE
        WHEN museum_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN museum_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_performing_arts_gross_floor_area_ft = CASE
        WHEN performing_arts_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN performing_arts_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_refrigerated_warehouse_gross_floor_area_ft = CASE
        WHEN refrigerated_warehouse_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN refrigerated_warehouse_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_restaurant_gross_floor_area_ft = CASE
        WHEN restaurant_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN restaurant_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_self_storage_facility_gross_floor_area_ft = CASE
        WHEN self_storage_facility_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN self_storage_facility_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_senior_living_community_gross_floor_area_ft = CASE
        WHEN senior_living_community_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN senior_living_community_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_social_meeting_hall_gross_floor_area_ft = CASE
        WHEN social_meeting_hall_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN social_meeting_hall_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_supermarket_grocery_gross_floor_area_ft = CASE
        WHEN supermarket_grocery_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN supermarket_grocery_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_wholesale_club_supercenter_gross_floor_area_ft = CASE
        WHEN wholesale_club_supercenter_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN wholesale_club_supercenter_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_wastewater_treatment_plant_gross_floor_area_ft = CASE
        WHEN wastewater_treatment_plant_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN wastewater_treatment_plant_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_worship_facility_gross_floor_area_ft = CASE
        WHEN worship_facility_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN worship_facility_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_adult_education_gross_floor_area_ft = CASE
        WHEN adult_education_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN adult_education_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_ambulatory_surgical_center_gross_floor_area_ft = CASE
        WHEN ambulatory_surgical_center_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN ambulatory_surgical_center_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_automobile_dealership_gross_floor_area_ft = CASE
        WHEN automobile_dealership_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN automobile_dealership_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_barracks_gross_floor_area_ft = CASE
        WHEN barracks_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN barracks_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_college_university_gross_floor_area_ft = CASE
        WHEN college_university_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN college_university_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_convention_center_gross_floor_area_ft = CASE
        WHEN convention_center_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN convention_center_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_courthouse_gross_floor_area_ft = CASE
        WHEN courthouse_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN courthouse_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_data_center_gross_floor_area_ft = CASE
        WHEN data_center_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN data_center_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END,
    temp_distribution_center_gross_floor_area_ft = CASE
        WHEN distribution_center_gross_floor_area_ft ~ '^[0-9.]+$'
        THEN distribution_center_gross_floor_area_ft::DOUBLE PRECISION
        ELSE NULL
    END;

-- Step 3: Drop original columns
ALTER TABLE your_table_name
DROP COLUMN hospital_general_medical_surgical_gross_floor_area_ft,
DROP COLUMN financial_office_gross_floor_area_ft,
DROP COLUMN retail_store_gross_floor_area_ft,
DROP COLUMN bank_branch_gross_floor_area_ft,
DROP COLUMN office_gross_floor_area_ft,
DROP COLUMN other_gross_floor_area_ft,
DROP COLUMN parking_gross_floor_area_ft,
DROP COLUMN enclosed_mall_gross_floor_area_ft,
DROP COLUMN energy_power_station_gross_floor_area_ft,
DROP COLUMN fitness_center_health_club_gym_gross_floor_area_ft,
DROP COLUMN food_sales_gross_floor_area_ft,
DROP COLUMN food_service_gross_floor_area_ft,
DROP COLUMN medical_office_gross_floor_area_ft,
DROP COLUMN urgent_care_clinic_other_outpatient_gross_floor_area_ft,
DROP COLUMN hotel_gross_floor_area_ft,
DROP COLUMN k_12_school_gross_floor_area_ft,
DROP COLUMN pre_school_daycare_gross_floor_area_ft,
DROP COLUMN laboratory_gross_floor_area_ft,
DROP COLUMN mailing_center_post_office_gross_floor_area_ft,
DROP COLUMN movie_theater_gross_floor_area_ft,
DROP COLUMN manufacturing_industrial_plant_gross_floor_area_ft,
DROP COLUMN multifamily_housing_gross_floor_area_ft,
DROP COLUMN residence_hall_dormitory_gross_floor_area_ft,
DROP COLUMN non_refrigerated_warehouse_gross_floor_area_ft,
DROP COLUMN museum_gross_floor_area_ft,
DROP COLUMN performing_arts_gross_floor_area_ft,
DROP COLUMN refrigerated_warehouse_gross_floor_area_ft,
DROP COLUMN restaurant_gross_floor_area_ft,
DROP COLUMN self_storage_facility_gross_floor_area_ft,
DROP COLUMN senior_living_community_gross_floor_area_ft,
DROP COLUMN social_meeting_hall_gross_floor_area_ft,
DROP COLUMN supermarket_grocery_gross_floor_area_ft,
DROP COLUMN wholesale_club_supercenter_gross_floor_area_ft,
DROP COLUMN wastewater_treatment_plant_gross_floor_area_ft,
DROP COLUMN worship_facility_gross_floor_area_ft,
DROP COLUMN adult_education_gross_floor_area_ft,
DROP COLUMN ambulatory_surgical_center_gross_floor_area_ft,
DROP COLUMN automobile_dealership_gross_floor_area_ft,
DROP COLUMN barracks_gross_floor_area_ft,
DROP COLUMN college_university_gross_floor_area_ft,
DROP COLUMN convention_center_gross_floor_area_ft,
DROP COLUMN courthouse_gross_floor_area_ft,
DROP COLUMN data_center_gross_floor_area_ft,
DROP COLUMN distribution_center_gross_floor_area_ft;

-- Step 4: Rename temporary columns to original column names
ALTER TABLE your_table_name
RENAME COLUMN temp_hospital_general_medical_surgical_gross_floor_area_ft TO hospital_general_medical_surgical_gross_floor_area_ft,
RENAME COLUMN temp_financial_office_gross_floor_area_ft TO financial_office_gross_floor_area_ft,
RENAME COLUMN temp_retail_store_gross_floor_area_ft TO retail_store_gross_floor_area_ft,
RENAME COLUMN temp_bank_branch_gross_floor_area_ft TO bank_branch_gross_floor_area_ft,
RENAME COLUMN temp_office_gross_floor_area_ft TO office_gross_floor_area_ft,
RENAME COLUMN temp_other_gross_floor_area_ft TO other_gross_floor_area_ft,
RENAME COLUMN temp_parking_gross_floor_area_ft TO parking_gross_floor_area_ft,
RENAME COLUMN temp_enclosed_mall_gross_floor_area_ft TO enclosed_mall_gross_floor_area_ft,
RENAME COLUMN temp_energy_power_station_gross_floor_area_ft TO energy_power_station_gross_floor_area_ft,
RENAME COLUMN temp_fitness_center_health_club_gym_gross_floor_area_ft TO fitness_center_health_club_gym_gross_floor_area_ft,
RENAME COLUMN temp_food_sales_gross_floor_area_ft TO food_sales_gross_floor_area_ft,
RENAME COLUMN temp_food_service_gross_floor_area_ft TO food_service_gross_floor_area_ft,
RENAME COLUMN temp_medical_office_gross_floor_area_ft TO medical_office_gross_floor_area_ft,
RENAME COLUMN temp_urgent_care_clinic_other_outpatient_gross_floor_area_ft TO urgent_care_clinic_other_outpatient_gross_floor_area_ft,
RENAME COLUMN temp_hotel_gross_floor_area_ft TO hotel_gross_floor_area_ft,
RENAME COLUMN temp_k_12_school_gross_floor_area_ft TO k_12_school_gross_floor_area_ft,
RENAME COLUMN temp_pre_school_daycare_gross_floor_area_ft TO pre_school_daycare_gross_floor_area_ft,
RENAME COLUMN temp_laboratory_gross_floor_area_ft TO laboratory_gross_floor_area_ft,
RENAME COLUMN temp_mailing_center_post_office_gross_floor_area_ft TO mailing_center_post_office_gross_floor_area_ft,
RENAME COLUMN temp_movie_theater_gross_floor_area_ft TO movie_theater_gross_floor_area_ft,
RENAME COLUMN temp_manufacturing_industrial_plant_gross_floor_area_ft TO manufacturing_industrial_plant_gross_floor_area_ft,
RENAME COLUMN temp_multifamily_housing_gross_floor_area_ft TO multifamily_housing_gross_floor_area_ft,
RENAME COLUMN temp_residence_hall_dormitory_gross_floor_area_ft TO residence_hall_dormitory_gross_floor_area_ft,
RENAME COLUMN temp_non_refrigerated_warehouse_gross_floor_area_ft TO non_refrigerated_warehouse_gross_floor_area_ft,
RENAME COLUMN temp_museum_gross_floor_area_ft TO museum_gross_floor_area_ft,
RENAME COLUMN temp_performing_arts_gross_floor_area_ft TO performing_arts_gross_floor_area_ft,
RENAME COLUMN temp_refrigerated_warehouse_gross_floor_area_ft TO refrigerated_warehouse_gross_floor_area_ft,
RENAME COLUMN temp_restaurant_gross_floor_area_ft TO restaurant_gross_floor_area_ft,
RENAME COLUMN temp_self_storage_facility_gross_floor_area_ft TO self_storage_facility_gross_floor_area_ft,
RENAME COLUMN temp_senior_living_community_gross_floor_area_ft TO senior_living_community_gross_floor_area_ft,
RENAME COLUMN temp_social_meeting_hall_gross_floor_area_ft TO social_meeting_hall_gross_floor_area_ft,
RENAME COLUMN temp_supermarket_grocery_gross_floor_area_ft TO supermarket_grocery_gross_floor_area_ft,
RENAME COLUMN temp_wholesale_club_supercenter_gross_floor_area_ft TO wholesale_club_supercenter_gross_floor_area_ft,
RENAME COLUMN temp_wastewater_treatment_plant_gross_floor_area_ft TO wastewater_treatment_plant_gross_floor_area_ft,
RENAME COLUMN temp_worship_facility_gross_floor_area_ft TO worship_facility_gross_floor_area_ft,
RENAME COLUMN temp_adult_education_gross_floor_area_ft TO adult_education_gross_floor_area_ft,
RENAME COLUMN temp_ambulatory_surgical_center_gross_floor_area_ft TO ambulatory_surgical_center_gross_floor_area_ft,
RENAME COLUMN temp_automobile_dealership_gross_floor_area_ft TO automobile_dealership_gross_floor_area_ft,
RENAME COLUMN temp_barracks_gross_floor_area_ft TO barracks_gross_floor_area_ft,
RENAME COLUMN temp_college_university_gross_floor_area_ft TO college_university_gross_floor_area_ft,
RENAME COLUMN temp_convention_center_gross_floor_area_ft TO convention_center_gross_floor_area_ft,
RENAME COLUMN temp_courthouse_gross_floor_area_ft TO courthouse_gross_floor_area_ft,
RENAME COLUMN temp_data_center_gross_floor_area_ft TO data_center_gross_floor_area_ft,
RENAME COLUMN temp_distribution_center_gross_floor_area_ft TO distribution_center_gross_floor_area_ft;

-- Step 5: Update columns to set NULL values to 0
UPDATE your_table_name
SET
    hospital_general_medical_surgical_gross_floor_area_ft = COALESCE(hospital_general_medical_surgical_gross_floor_area_ft, 0),
    financial_office_gross_floor_area_ft = COALESCE(financial_office_gross_floor_area_ft, 0),
    retail_store_gross_floor_area_ft = COALESCE(retail_store_gross_floor_area_ft, 0),
    bank_branch_gross_floor_area_ft = COALESCE(bank_branch_gross_floor_area_ft, 0),
    office_gross_floor_area_ft = COALESCE(office_gross_floor_area_ft, 0),
    other_gross_floor_area_ft = COALESCE(other_gross_floor_area_ft, 0),
    parking_gross_floor_area_ft = COALESCE(parking_gross_floor_area_ft, 0),
    enclosed_mall_gross_floor_area_ft = COALESCE(enclosed_mall_gross_floor_area_ft, 0),
    energy_power_station_gross_floor_area_ft = COALESCE(energy_power_station_gross_floor_area_ft, 0),
    fitness_center_health_club_gym_gross_floor_area_ft = COALESCE(fitness_center_health_club_gym_gross_floor_area_ft, 0),
    food_sales_gross_floor_area_ft = COALESCE(food_sales_gross_floor_area_ft, 0),
    food_service_gross_floor_area_ft = COALESCE(food_service_gross_floor_area_ft, 0),
    medical_office_gross_floor_area_ft = COALESCE(medical_office_gross_floor_area_ft, 0),
    urgent_care_clinic_other_outpatient_gross_floor_area_ft = COALESCE(urgent_care_clinic_other_outpatient_gross_floor_area_ft, 0),
    hotel_gross_floor_area_ft = COALESCE(hotel_gross_floor_area_ft, 0),
    k_12_school_gross_floor_area_ft = COALESCE(k_12_school_gross_floor_area_ft, 0),
    pre_school_daycare_gross_floor_area_ft = COALESCE(pre_school_daycare_gross_floor_area_ft, 0),
    laboratory_gross_floor_area_ft = COALESCE(laboratory_gross_floor_area_ft, 0),
    mailing_center_post_office_gross_floor_area_ft = COALESCE(mailing_center_post_office_gross_floor_area_ft, 0),
    movie_theater_gross_floor_area_ft = COALESCE(movie_theater_gross_floor_area_ft, 0),
    manufacturing_industrial_plant_gross_floor_area_ft = COALESCE(manufacturing_industrial_plant_gross_floor_area_ft, 0),
    multifamily_housing_gross_floor_area_ft = COALESCE(multifamily_housing_gross_floor_area_ft, 0),
    residence_hall_dormitory_gross_floor_area_ft = COALESCE(residence_hall_dormitory_gross_floor_area_ft, 0),
    non_refrigerated_warehouse_gross_floor_area_ft = COALESCE(non_refrigerated_warehouse_gross_floor_area_ft, 0),
    museum_gross_floor_area_ft = COALESCE(museum_gross_floor_area_ft, 0),
    performing_arts_gross_floor_area_ft = COALESCE(performing_arts_gross_floor_area_ft, 0),
    refrigerated_warehouse_gross_floor_area_ft = COALESCE(refrigerated_warehouse_gross_floor_area_ft, 0),
    restaurant_gross_floor_area_ft = COALESCE(restaurant_gross_floor_area_ft, 0),
    self_storage_facility_gross_floor_area_ft = COALESCE(self_storage_facility_gross_floor_area_ft, 0),
    senior_living_community_gross_floor_area_ft = COALESCE(senior_living_community_gross_floor_area_ft, 0),
    social_meeting_hall_gross_floor_area_ft = COALESCE(social_meeting_hall_gross_floor_area_ft, 0),
    supermarket_grocery_gross_floor_area_ft = COALESCE(supermarket_grocery_gross_floor_area_ft, 0),
    wholesale_club_supercenter_gross_floor_area_ft = COALESCE(wholesale_club_supercenter_gross_floor_area_ft, 0),
    wastewater_treatment_plant_gross_floor_area_ft = COALESCE(wastewater_treatment_plant_gross_floor_area_ft, 0),
    worship_facility_gross_floor_area_ft = COALESCE(worship_facility_gross_floor_area_ft, 0),
    adult_education_gross_floor_area_ft = COALESCE(adult_education_gross_floor_area_ft, 0),
    ambulatory_surgical_center_gross_floor_area_ft = COALESCE(ambulatory_surgical_center_gross_floor_area_ft, 0),
    automobile_dealership_gross_floor_area_ft = COALESCE(automobile_dealership_gross_floor_area_ft, 0),
    barracks_gross_floor_area_ft = COALESCE(barracks_gross_floor_area_ft, 0),
    college_university_gross_floor_area_ft = COALESCE(college_university_gross_floor_area_ft, 0),
    convention_center_gross_floor_area_ft = COALESCE(convention_center_gross_floor_area_ft, 0),
    courthouse_gross_floor_area_ft = COALESCE(courthouse_gross_floor_area_ft, 0),
    data_center_gross_floor_area_ft = COALESCE(data_center_gross_floor_area_ft, 0),
    distribution_center_gross_floor_area_ft = COALESCE(distribution_center_gross_floor_area_ft, 0);
