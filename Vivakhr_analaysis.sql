/*===================================================================================
          SQL FINAL GROUP PROJECT
=====================================================================================
Description: The HR Manager of VivaK, a successful fashion retail chain based in Southlake, Texas, has recruited a team of analysts to develop an Online Analytical Processing (OLAP) 
Database for enhancing their HR analytics. Due to the current management's data being stored in multiple formats and containing anomalies, the team is tasked with 
analyzing the data, creating a schema, and executing specific queries. */ 
/***************   BEGINNNING OF VIVAKHR SCHEMA CREATION     *****************/
-- Drop Schema vivakhr if it already exists.
DROP SCHEMA IF EXISTS vivakhr;

-- Create schema vivakhr.
CREATE SCHEMA IF NOT EXISTS vivakhr;

-- Select the 'vivakhr' database for subsequent operations.
USE vivakhr;

/* Create a Schema based on VivaK_Data_Model and call it VivaKHR. */
/* You must include all the statements in your SQL file and only the important statements/ outputs in your presentation. */

-- Create a new table 'regions' if it does not already exist.
CREATE TABLE IF NOT EXISTS regions (
     region_id   INT,						
     region_name VARCHAR(200) NOT NULL,	
     PRIMARY KEY (region_id)				--  Sets 'region_id' as the primary key for the 'regions' table.
  ); 

-- Create a new table 'countries' if it does not already exist.
CREATE TABLE IF NOT EXISTS countries (
     country_id INT AUTO_INCREMENT,			
     country_code VARCHAR(3),				
     country_name VARCHAR(150) NOT NULL,	
     region_id    INT,						
	 
     PRIMARY KEY (country_id),				--  Sets 'country_id' as the primary key for the 'countries' table.
     CONSTRAINT countries_region_fk
		FOREIGN KEY (region_id) REFERENCES regions (region_id)    -- Establish a foreign key relationship between 'region_id' in the 'countries' table and 'region_id' in the 'regions' table.
			ON UPDATE RESTRICT 					-- 'ON UPDATE RESTRICT' prevents 'region_id' updates in 'regions'.
			ON DELETE CASCADE				    -- 'ON DELETE CASCADE' auto deletes related child 'countries' table observations. 
  ); 

-- Create a new table 'locations' if it does not already exist.
CREATE TABLE IF NOT EXISTS locations (
     location_id    INT,					
     location_code  INT,					
     street_address VARCHAR(300),			
     postal_code    VARCHAR(8),				
     city           VARCHAR(30),			
     state_province VARCHAR(30),			
     country_id     INT,					
	 
     PRIMARY KEY (location_id),				--  Sets 'location_id' as the primary key for the 'locations' table.
     CONSTRAINT locations_country_fk
		FOREIGN KEY (country_id) REFERENCES COUNTRIES (country_id)	 -- Establish a foreign key relationship between 'country_id' in the 'locations' table and 'country_id' in the 'countries' table.
			ON UPDATE RESTRICT 					-- 'ON UPDATE RESTRICT' prevents 'country_id' updates in 'countries'.
			ON DELETE CASCADE					-- 'ON DELETE CASCADE' auto deletes related child 'locations' table observations. 
  ); 

-- Create a new table 'departments' if it does not already exist.
CREATE TABLE IF NOT EXISTS departments (
     department_id   INT,					
     department_name VARCHAR(50),			
       
     PRIMARY KEY (department_id)			--  Sets 'department_id' as the primary key for the 'departments' table.
  ); 

-- Create a new table 'jobs' if it does not already exist.
CREATE TABLE IF NOT EXISTS jobs (
     job_id        INT,						
     job_title     VARCHAR(100),			
     min_salary    DOUBLE(10,2),			
     max_salary    DOUBLE(10,2),			
     department_id INT,						
	 
     PRIMARY KEY (job_id),					--  Sets 'job_id' as the primary key for the 'jobs' table.
     CONSTRAINT jobs_dep_fk
		FOREIGN KEY (department_id) REFERENCES departments(department_id) 	-- Establish a foreign key relationship between 'department_id' in the 'jobs' table and 'department_id' in the 'departments' table.
			ON UPDATE RESTRICT 					-- 'ON UPDATE RESTRICT' prevents 'department_id' updates in 'departments'.			
			ON DELETE CASCADE					-- 'ON DELETE CASCADE' auto deletes related child 'jobs' table observations. 		
  ); 

-- Create a new table 'employees' if it does not already exist.
CREATE TABLE IF NOT EXISTS employees (
     employee_id INT,						
     first_name VARCHAR(50) NOT NULL,		
     last_name VARCHAR(50) NOT NULL,		
     email VARCHAR(100) NOT NULL,			
     phone_number VARCHAR(17),				
     job_id INT,							
     location_id INT,						
     salary DOUBLE(10,2),					
     report_to INT,							
     hire_date DATE,						
     experience_at_vivak INT,				
     last_performance_rating DOUBLE,		
     salary_after_increment DOUBLE(10,2),	
     annual_dependent_benefit DOUBLE(10,2),	
     
     PRIMARY KEY (employee_id),				--  Sets 'employee_id' as the primary key for the 'employees' table.
     CONSTRAINT employees_job_fk 
		FOREIGN KEY (job_id) REFERENCES jobs (job_id)	-- Establish a foreign key relationship between 'job_id' in the 'employees' table and 'job_id' in the 'jobs' table.
			ON UPDATE RESTRICT					-- 'ON UPDATE RESTRICT' prevents 'job_id' updates in 'jobs'.  
			ON DELETE CASCADE,					-- 'ON DELETE CASCADE' auto deletes related child 'employees' table observations. 		
     CONSTRAINT employees_loc_fk 
		FOREIGN KEY (location_id) REFERENCES locations (location_id)	-- Establish a foreign key relationship between 'location_id' in the 'employees' table and 'location_id' in the 'locations' table. 
			ON UPDATE RESTRICT					-- 'ON UPDATE RESTRICT' prevents 'location_id' updates in 'locations'.
			ON DELETE CASCADE, 					-- 'ON DELETE CASCADE' auto deletes related child 'employees' table observations. 		
     CONSTRAINT employees_emp_fk 
		FOREIGN KEY (report_to) REFERENCES employees (employee_id)	-- Establish a foreign key relationship between 'employee_id' in the 'employees' table and 'employee_id' in the 'employees' table.
			ON UPDATE RESTRICT 					-- 'ON UPDATE RESTRICT' prevents 'employee_id' updates in 'employees'.
			ON DELETE CASCADE,					-- 'ON DELETE CASCADE' auto deletes related 'employees' table observations. 		
     CONSTRAINT CHECK(last_performance_rating between 0 and 10)		-- Check constraint to enforces last_performance_rating values are between 0 and 10.
);

-- Create a new table 'dependents' if it does not already exist.
CREATE TABLE IF NOT EXISTS dependents (
     dependent_id INT auto_increment,		
     first_name   VARCHAR(50) NOT NULL,		
     last_name    VARCHAR(50) NOT NULL,		
     relationship VARCHAR(10) NOT NULL,		
     employee_id  INT,						
	 
     PRIMARY KEY (dependent_id),			--  Sets 'dependent_id' as the primary key for the 'dependents' table.
     CONSTRAINT dependents_emp_fk
		FOREIGN KEY (employee_id) REFERENCES employees(employee_id)	-- Establish a foreign key relationship between 'employee_id' in the 'dependents' table and 'employee_id' in the 'employees' table.
			ON UPDATE RESTRICT					-- 'ON UPDATE RESTRICT' prevents 'employee_id' updates in 'employees'.
			ON DELETE CASCADE					-- 'ON DELETE CASCADE' auto deletes related child 'dependents' table observations. 	
  ); 
 
-- Inserting Data into vivakhr schema tables.
USE vivakhr;
-- Insert data into the 'regions' table by selecting records from the 'vivakdump.regions' table.
INSERT INTO regions(region_id, region_name)
      SELECT region_id, region_name FROM vivakdump.regions; 

-- Insert data into the 'countries' table by selecting records from the 'vivakdump.countries' table.       
INSERT INTO countries(country_code, country_name,region_id)
      SELECT country_id, country_name, region_id FROM vivakdump.countries;
      
-- Insert data into the 'locations' table by selecting records from the 'vivakdump.locations' table, 'countries' table.      
INSERT INTO locations(location_id, location_code, street_address, postal_code, city, state_province, country_id)
      SELECT location_id, location_code, street_address, postal_code, city, state_province, c.country_id	
		FROM vivakdump.locations AS l inner join countries AS c on (c.country_code = l.country_id);

-- Insert data into the 'departments' table by selecting records from the 'vivakdump.departments' table.
INSERT INTO departments(department_id, department_name)
      SELECT department_id, department_name FROM vivakdump.departments;

-- Insert distinct data into the 'jobs' table by selecting records from the 'vivakdump.jobs' table, 'vivakdump.ORGSTRUCTURE' table, 'vivakdump.DEPARTMENTS' table.
INSERT INTO jobs(job_id, job_title, min_salary, max_salary, department_id)		
	SELECT DISTINCT o.job_id, o.job_title, o.min_salary, o.max_salary, d.department_id 
    FROM vivakdump.ORGSTRUCTURE AS o                                 			-- Right join on orgstructure table to import missing job id's. 
		INNER JOIN vivakdump.DEPARTMENTS AS d USING (department_name);   		-- Inner join on departments table to retrieve department_id. 

-- Insert distinct data into the 'employees' table by selecting records from the 'vivakdump.employees', 'jobs' table, 'locations' table, 'countries' table.
INSERT INTO employees(employee_id, first_name, last_name, email, phone_number, job_id, salary, report_to, hire_date, location_id)
      SELECT DISTINCT e.employee_id, e.first_name, e.last_name, e.email,
                      concat(CASE														
                               WHEN country_code IN ( 'US', 'CAN' ) THEN '+001-'    	
                               WHEN country_code = 'UK' THEN '+044-'
                               WHEN country_code = 'DE' THEN '+045-'
                               ELSE '+000-'
                             END, replace(e.phone_number, '.', '-')) AS phone_number,   
                      e.job_id,
                      e.salary,
                      NULL,
                      date(e.hire_date) AS hire_date,                                   
                      e.department_id   AS location_id                                  
        FROM vivakdump.employees AS e
             INNER JOIN jobs AS j using (job_id)										
             INNER JOIN locations AS l													
                     ON ( e.department_id = l.location_id )                             
             INNER JOIN countries AS c using (country_id); 								

-- Insert data into the 'dependents' table by selecting records from the 'vivakdump.dependent' table, 'employees' table.      
INSERT INTO dependents(first_name, last_name, relationship, employee_id)
      SELECT d.first_name, d.last_name, d.relationship, d.employee_id
        FROM vivakdump.dependent AS d
             INNER JOIN employees AS e using (employee_id);       -- Inner join with 'employees' table to ignore the orphan records from 'dependents' tables.

-- Retaining Select queries consistently across the SQL script for convenient querying.              
select * from regions; 
select * from countries; 
select * from locations;
select * from departments;
select * from jobs;
select * from employees;
select * from dependents;
 
-- Fill up the report_to column by analyzing the available data.
update employees e
	join vivakdump.orgstructure j using (job_id) 
	join employees mgr on (reports_to = mgr.job_id
	and case 
			when j.reports_to = 1 then e.location_id 
            else mgr.location_id  end  = e.location_id)
	set e.report_to = mgr.employee_id;


--  Devise a strategy to fill in the missing entries in the salary column. Justify your answers and state your assumptions.
UPDATE employees as e
	JOIN jobs as j using (job_id)
	SET e.salary = (j.min_salary + j.max_salary) / 2
	WHERE e.salary = 0;

UPDATE employees
	SET experience_at_VivaK = TIMESTAMPDIFF(month, hire_date, CURRENT_DATE);	-- Calculating the difference in months between hire date and current date.
select * from employees;


UPDATE employees e		
	LEFT JOIN jobs j ON e.job_id = j.job_id
	SET e.salary_after_increment = LEAST(e.salary + ((1 + (0.01 * e.experience_at_VivaK) + 
		CASE										
            WHEN e.last_performance_rating >= 9 THEN 0.15
            WHEN e.last_performance_rating >= 8 THEN 0.12
            WHEN e.last_performance_rating >= 7 THEN 0.10
            WHEN e.last_performance_rating >= 6 THEN 0.08
            WHEN e.last_performance_rating >= 5 THEN 0.05
            ELSE 0.02
        END
    ) * e.salary) / 100,
    j.max_salary
);
    
-- Annual_dependent_benefit: Calculate the annual dependent benefit per dependent (in USD) and update the column as per the table below
/*	Title				Dependent benefit per dependent
	Executives			0.2 * annual salary
	Managers			0.15 * annual salary
	Other Employees		0.05 * annual salary		*/

UPDATE employees AS e		
SET e.annual_dependent_benefit = (
    SELECT 				 
        CASE			
            WHEN dp.department_name LIKE '%Executive%' THEN e.salary * 0.2 * (SELECT COUNT(d.dependent_id) FROM dependents AS d WHERE d.employee_id = e.employee_id)
            WHEN j.job_title LIKE '%Manager%' THEN e.salary * 0.15 * (SELECT COUNT(d.dependent_id) FROM dependents AS d WHERE d.employee_id = e.employee_id)
            ELSE e.salary * 0.05 * (SELECT COUNT(d.dependent_id) FROM dependents AS d WHERE d.employee_id = e.employee_id)
        END
    FROM jobs AS j
    LEFT JOIN departments AS dp ON j.department_id = dp.department_id
    WHERE e.job_id = j.job_id
);

-- Updating email field for all records in 'employees' table.
UPDATE employees
	SET email = CONCAT(SUBSTRING_INDEX(email, '@', 1), '@vivaK.com');	-- Extracting the part of email before @ symbol and appending the domain '@vivak.com' to the extracted username
