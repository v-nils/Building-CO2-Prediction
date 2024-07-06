/* Table Pluto contains lot information such as num of buildings, num of floors, lot and building depth... */
/* Table energy data, contains energy information of lots */
drop table if exists energy_pluto_data;
create table energy_pluto_data as (
    WITH energy as (
        select
            regexp_replace(unnest(string_to_array(e.nyc_building_identification_number_bin, ';')), '[^0-9]', '', 'g') as bin,
            regexp_replace(unnest(string_to_array(e.bbl, ';')), '[^0-9]', '', 'g') as e_bbl,
            occupancy,
            hospital_general_medical_surgical_gross_floor_area_ft,
            financial_office_gross_floor_area_ft,
            retail_store_gross_floor_area_ft,
            bank_branch_gross_floor_area_ft,
            office_gross_floor_area_ft,
            other_gross_floor_area_ft,
            parking_gross_floor_area_ft,
            enclosed_mall_gross_floor_area_ft,
            energy_power_station_gross_floor_area_ft,
            fitness_center_health_club_gym_gross_floor_area_ft,
            food_sales_gross_floor_area_ft,
            food_service_gross_floor_area_ft,
            medical_office_gross_floor_area_ft,
            urgent_care_clinic_other_outpatient_gross_floor_area_ft,
            hotel_gross_floor_area_ft,
            k_12_school_gross_floor_area_ft,
            pre_school_daycare_gross_floor_area_ft,
            laboratory_gross_floor_area_ft,
            mailing_center_post_office_gross_floor_area_ft,
            movie_theater_gross_floor_area_ft,
            manufacturing_industrial_plant_gross_floor_area_ft,
            multifamily_housing_gross_floor_area_ft,
            residence_hall_dormitory_gross_floor_area_ft,
            non_refrigerated_warehouse_gross_floor_area_ft,
            museum_gross_floor_area_ft,
            performing_arts_gross_floor_area_ft,
            refrigerated_warehouse_gross_floor_area_ft,
            restaurant_gross_floor_area_ft,
            self_storage_facility_gross_floor_area_ft,
            senior_living_community_gross_floor_area_ft,
            social_meeting_hall_gross_floor_area_ft,
            supermarket_grocery_gross_floor_area_ft,
            wholesale_club_supercenter_gross_floor_area_ft,
            wastewater_treatment_plant_gross_floor_area_ft,
            worship_facility_gross_floor_area_ft,
            adult_education_gross_floor_area_ft,
            ambulatory_surgical_center_gross_floor_area_ft,
            automobile_dealership_gross_floor_area_ft,
            barracks_gross_floor_area_ft,
            college_university_gross_floor_area_ft,
            convention_center_gross_floor_area_ft,
            courthouse_gross_floor_area_ft,
            data_center_gross_floor_area_ft,
            distribution_center_gross_floor_area_ft,
            weather_normalized_site_energy_use_kbtu,
            weather_normalized_site_eui_kbtu_ft,
            total_ghg_emissions_metric_tons_co2e,
            total_ghg_emissions_intensity_kgco2e_ft

        FROM energy_data as e),

    pluto as (
        select
            regexp_replace(unnest(string_to_array(cast(p.bbl as text), ';')), '[^0-9]', '', 'g') as p_bbl,
            lotarea,
            bldgarea,
            comarea,
            resarea,
            officearea,
            retailarea,
            garagearea,
            strgearea,
            factryarea,
            otherarea,
            numbldgs,
            numfloors,
            unitstotal,
            lotfront,
            lotdepth,
            bldgfront,
            bldgdepth,
            proxcode,
            irrlotcode,
            yearbuilt,
            case
                when not yearalter2 = 0 then yearalter2
                when yearalter1 > yearalter2 then yearalter1
                else yearbuilt
            end as yearaltered,
            lottype,
            assessland,
            assesstot,
            builtfar,
            residfar,
            commfar,
            facilfar,
            shape_area,
            shape,
            cd
        FROM map_pluto as p)

    select *
    from (
        select
                case
                    when energy.e_bbl is null then pluto.p_bbl
                    else energy.e_bbl
                end as bbl,
                lotarea,
                bldgarea,
                comarea,
                resarea,
                officearea,
                retailarea,
                garagearea,
                strgearea,
                factryarea,
                otherarea,
                numbldgs,
                numfloors,
                unitstotal,
                lotfront,
                lotdepth,
                bldgfront,
                bldgdepth,
                proxcode,
                irrlotcode,
                yearbuilt,
                yearaltered,
                assessland,
                assesstot,
                lottype,
                builtfar,
                residfar,
                commfar,
                facilfar,
                shape_area,
                hospital_general_medical_surgical_gross_floor_area_ft,
                financial_office_gross_floor_area_ft,
                retail_store_gross_floor_area_ft,
                bank_branch_gross_floor_area_ft,
                office_gross_floor_area_ft,
                other_gross_floor_area_ft,
                parking_gross_floor_area_ft,
                enclosed_mall_gross_floor_area_ft,
                energy_power_station_gross_floor_area_ft,
                fitness_center_health_club_gym_gross_floor_area_ft,
                food_sales_gross_floor_area_ft,
                food_service_gross_floor_area_ft,
                medical_office_gross_floor_area_ft,
                urgent_care_clinic_other_outpatient_gross_floor_area_ft,
                hotel_gross_floor_area_ft,
                k_12_school_gross_floor_area_ft,
                pre_school_daycare_gross_floor_area_ft,
                laboratory_gross_floor_area_ft,
                mailing_center_post_office_gross_floor_area_ft,
                movie_theater_gross_floor_area_ft,
                manufacturing_industrial_plant_gross_floor_area_ft,
                multifamily_housing_gross_floor_area_ft,
                residence_hall_dormitory_gross_floor_area_ft,
                non_refrigerated_warehouse_gross_floor_area_ft,
                museum_gross_floor_area_ft,
                performing_arts_gross_floor_area_ft,
                refrigerated_warehouse_gross_floor_area_ft,
                restaurant_gross_floor_area_ft,
                self_storage_facility_gross_floor_area_ft,
                senior_living_community_gross_floor_area_ft,
                social_meeting_hall_gross_floor_area_ft,
                supermarket_grocery_gross_floor_area_ft,
                wholesale_club_supercenter_gross_floor_area_ft,
                wastewater_treatment_plant_gross_floor_area_ft,
                worship_facility_gross_floor_area_ft,
                adult_education_gross_floor_area_ft,
                ambulatory_surgical_center_gross_floor_area_ft,
                automobile_dealership_gross_floor_area_ft,
                barracks_gross_floor_area_ft,
                college_university_gross_floor_area_ft,
                convention_center_gross_floor_area_ft,
                courthouse_gross_floor_area_ft,
                data_center_gross_floor_area_ft,
                distribution_center_gross_floor_area_ft,
                weather_normalized_site_energy_use_kbtu,
                weather_normalized_site_eui_kbtu_ft,
                total_ghg_emissions_metric_tons_co2e,
                total_ghg_emissions_intensity_kgco2e_ft,
                shape as geom,
                cd
        from energy
        full outer join pluto
        on energy.e_bbl = pluto.p_bbl) as a
    join median_incomes_filtered on fips = cd
    where timeframe = 2021 and household_type = 'All Households');


alter table energy_pluto_data add column bef_id serial primary key;
create index energy_pluto_data_bbl_inner_idx on energy_pluto_data(bbl);
create index energy_pluto_data_inner_geom_idx on energy_pluto_data using gist(geom);

delete from energy_pluto_data
where bbl in (select bbl from energy_pluto_data group by bbl having count(*) > 1);


select count(*)
from energy_pluto_data;

drop table if exists processing_dataset;
create table processing_dataset as (
    with epluto as (
        select
            *,
            energy_pluto_data.bbl as e_bbl
        from energy_pluto_data join bldgs_full_3d on energy_pluto_data.bbl = bldgs_full_3d.bbl::text)
    select
        epluto.e_bbl as bbl,
        epluto.median_income,
        epluto.lotarea,
        epluto.bldgarea,
        epluto.numbldgs,
        epluto.numfloors,
        epluto.unitstotal,
        epluto.lotfront,
        epluto.lotdepth,
        epluto.bldgfront,
        epluto.bldgdepth,
        case
            when epluto.irrlotcode = 'N' then 0
            when epluto.irrlotcode = 'Y' then 1
            else null
        end as irrlotcode,
        epluto.lottype,
        epluto.assessland,
        epluto.assesstot,
        epluto.proxcode,
        epluto.yearbuilt,
        epluto.yearaltered,
        epluto.builtfar,
        epluto.residfar,
        epluto.commfar,
        epluto.facilfar,
        epluto.shape_area,
        epluto.z_min,
        epluto.z_max,
        epluto.z_mean,
        epluto.sarea,
        epluto.volume,
        epluto.comarea,
        epluto.resarea,
        epluto.officearea,
        epluto.retailarea,
        epluto.garagearea,
        epluto.strgearea,
        epluto.factryarea,
        epluto.otherarea,
        epluto.hospital_general_medical_surgical_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as hospital_general_medical_surgical_gross_floor_area_ratio,
        epluto.financial_office_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as financial_office_gross_floor_area_ratio,
        epluto.retail_store_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as retail_store_gross_floor_area_ratio,
        epluto.bank_branch_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as bank_branch_gross_floor_area_ratio,
        epluto.office_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as office_gross_floor_area_ratio,
        epluto.other_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as other_gross_floor_area_ratio,
        epluto.parking_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as parking_gross_floor_area_ratio,
        epluto.enclosed_mall_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as enclosed_mall_gross_floor_area_ratio,
        epluto.energy_power_station_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as energy_power_station_gross_floor_area_ratio,
        epluto.fitness_center_health_club_gym_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as fitness_center_health_club_gym_gross_floor_area_ratio,
        epluto.food_sales_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as food_sales_gross_floor_area_ratio,
        epluto.food_service_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as food_service_gross_floor_area_ratio,
        epluto.medical_office_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as medical_office_gross_floor_area_ratio,
        epluto.urgent_care_clinic_other_outpatient_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as urgent_care_clinic_other_outpatient_gross_floor_area_ratio,
        epluto.hotel_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as hotel_gross_floor_area_ratio,
        epluto.k_12_school_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as k_12_school_gross_floor_area_ratio,
        epluto.pre_school_daycare_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as pre_school_daycare_gross_floor_area_ratio,
        epluto.laboratory_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as laboratory_gross_floor_area_ratio,
        epluto.mailing_center_post_office_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as mailing_center_post_office_gross_floor_area_ratio,
        epluto.movie_theater_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as movie_theater_gross_floor_area_ratio,
        epluto.manufacturing_industrial_plant_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as manufacturing_industrial_plant_gross_floor_area_ratio,
        epluto.multifamily_housing_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as multifamily_housing_gross_floor_area_ratio,
        epluto.residence_hall_dormitory_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as residence_hall_dormitory_gross_floor_area_ratio,
        epluto.non_refrigerated_warehouse_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as non_refrigerated_warehouse_gross_floor_area_ratio,
        epluto.museum_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as museum_gross_floor_area_ratio,
        epluto.performing_arts_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as performing_arts_gross_floor_area_ratio,
        epluto.refrigerated_warehouse_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as refrigerated_warehouse_gross_floor_area_ratio,
        epluto.restaurant_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as restaurant_gross_floor_area_ratio,
        epluto.self_storage_facility_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as self_storage_facility_gross_floor_area_ratio,
        epluto.senior_living_community_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as senior_living_community_gross_floor_area_ratio,
        epluto.social_meeting_hall_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as social_meeting_hall_gross_floor_area_ratio,
        epluto.supermarket_grocery_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as supermarket_grocery_gross_floor_area_ratio,
        epluto.wholesale_club_supercenter_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as wholesale_club_supercenter_gross_floor_area_ratio,
        epluto.wastewater_treatment_plant_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as wastewater_treatment_plant_gross_floor_area_ratio,
        epluto.worship_facility_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as worship_facility_gross_floor_area_ratio,
        epluto.adult_education_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as adult_education_gross_floor_area_ratio,
        epluto.ambulatory_surgical_center_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as ambulatory_surgical_center_gross_floor_area_ratio,
        epluto.automobile_dealership_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as automobile_dealership_gross_floor_area_ratio,
        epluto.barracks_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as barracks_gross_floor_area_ratio,
        epluto.college_university_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as college_university_gross_floor_area_ratio,
        epluto.convention_center_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as convention_center_gross_floor_area_ratio,
        epluto.courthouse_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as courthouse_gross_floor_area_ratio,
        epluto.data_center_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as data_center_gross_floor_area_ratio,
        epluto.distribution_center_gross_floor_area_ft / epluto.bldgarea::DOUBLE PRECISION as distribution_center_gross_floor_area_ratio,
        epluto.weather_normalized_site_energy_use_kbtu,
        epluto.weather_normalized_site_eui_kbtu_ft,
        epluto.total_ghg_emissions_metric_tons_co2e,
        epluto.total_ghg_emissions_intensity_kgco2e_ft
    from epluto);

delete from processing_dataset
where bbl in (select bbl from processing_dataset group by bbl having count(*) > 1);

select count(*)
from processing_dataset;




