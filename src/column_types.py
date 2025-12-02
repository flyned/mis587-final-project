
location_columns = [
    'Property Address', 'Market Name', 'Submarket Name', 'Submarket Cluster', 'City', 'State',
    'Zip', 'County Name', 'Building Park', 'Continent', 'Country', 'Latitude', 'Longitude', 'Subcontinent',
    'Cross Street'
]

categorical_columns = [
    'Property Type', 'Building Class', 'Building Status', 'Secondary Type', 'For Sale Status','Tenancy', 'In SFHA',
    'Floodplain Area', 'Sewer', 'Collateral Type', 'Condo', 'Constr Status', 'Construction Material', 'Direct Services',
    'Fema Flood Zone', 'Flood Risk Area', 'Gas', 'Has Lab Space', 'Heating', 'Scale', 'Services', 'Sprinklers', 'Water',
    'Zoning'
]

numerical_columns = [
    'Star Rating', 'RBA', 'Total Available Space (SF)', 'Rent/SF/Yr', 'For Sale Price', 'Last Sale Price',
    'Percent Leased', 'Typical Floor Size', 'Parking Ratio', 'Ceiling Ht', 'Number Of Cranes',
    'Number Of Loading Docks', 'Drive Ins', 'Average Weighted Rent', 'Avg Rent-Direct (Industrial)',
    'Avg Rent-Direct (Office)', 'Avg Rent-Direct (Retail)', 'Avg Rent-Sublet (Industrial)', 'Avg Rent-Sublet (Office)',
    'Avg Rent-Sublet (Retail)', 'Building Operating Expenses', 'Building Tax Expenses', 'Coworking Available Space',
    'Days On Market', 'Direct Available Space', 'Direct Vacant Space', 'Land Area (AC)', 'Land Area (SF)',
    'Max Building Contiguous Space', 'Max Floor Contiguous Space', 'Number Of Elevators', 'Number Of Parking Spaces',
    'Number Of Stories', 'Office Space', 'Origination Amount', 'Smallest Available Space', 'Sublet Available Space',
    'Sublet Vacant Space', 'Taxes Per SF', 'Taxes Total', 'Total Relet Space (SF)', 'Total Sublet Space (SF)',
    'Total Vacant Avail Relet Space (SF)', 'Total Vacant Avail Sublet Space (SF)', 'Total Vacant Available',
    'Vacancy %',
]

numerical_potential_bin = [
    'Star Rating', 'Number Of Elevators', 'Number Of Stories'
]

numerical_needs_cleaning_columns = [
    'Rent/SF/Yr', 'Ceiling Ht', 'Number Of Cranes', 'Drive Ins', 'Building Operating Expenses', 'Building Tax Expenses',
]

date_columns = [
    'Last Sale Date', 'FEMA Map Date', 'Construction Begin', 'Maturity Date', 'Origination Date'
]

year_columns = [
    'Year Built', 'Year Renovated', 'Tax Year'
]

# ----------------------------
# Columns that will be dropped
# ----------------------------
not_relevant_columns = [
    'Property Name', 'Leasing Company Name', 'Leasing Company Contact', 'Sale Company Name', 'Sale Company Contact',
    'FEMA Map Identifier', 'FIRM ID', 'FIRM Panel Number', 'Architect Name', 'Developer Name', 'Fund Name',
    'Leasing Company Address', 'Leasing Company City State Zip', 'Leasing Company Fax', 'Leasing Company Phone',
    'Originator', 'Owner Address', 'Owner City State Zip', 'Owner Contact', 'Owner Name', 'Owner Phone',
    'Parcel Number 1(Min)', 'Parcel Number 2(Max)', 'Parent Company', 'Primary Agent Name', 'Primary Corridors',
    'Property Manager Address', 'Property Manager City State Zip', 'Property Manager Contact', 'Property Manager Name',
    'Property Manager Phone', 'PropertyID', 'Recorded Owner Address', 'Recorded Owner City State Zip',
    'Recorded Owner Contact', 'Recorded Owner Name', 'Recorded Owner Phone', 'Restaurant', 'Sale Company Address',
    'Sale Company City State Zip', 'Sale Company Fax', 'Sale Company Phone', 'Sales Company', 'Sales Contact',
    'Sales Contact Phone', 'True Owner Address', 'True Owner City State Zip', 'True Owner Contact', 'True Owner Name',
    'True Owner Phone',
]

not_relevant_residential_columns = [
    '% 1-Bed', '% 2-Bed', '% 3-Bed', '% 4-Bed', '% Studios', 'Affordable Type', 'All-Inclusive', 'All-Suites',
    'Avg Asking/Bed', 'Avg Asking/SF', 'Avg Asking/Unit', 'Brand', 'For Sale Price/Room',
    'Four Bedroom Asking Rent/Bed', 'Four Bedroom Asking Rent/SF', 'Four Bedroom Asking Rent/Unit',
    'Four Bedroom Avg SF', 'Four Bedroom Concessions %', 'Four Bedroom Effective Rent/Bed',
    'Four Bedroom Effective Rent/SF', 'Four Bedroom Effective Rent/Unit', 'Four Bedroom Vacancy %',
    'Four Bedroom Vacant Units', 'Hotel Class', 'Hotel Grade', 'Hotel Location Type', 'Hotel Open Date',
    'Hotel Operator', 'Number Of 1 Bedrooms Units', 'Number Of 2 Bedrooms Units', 'Number Of 3 Bedrooms Units',
    'Number Of 4 Bedrooms Units', 'Number of Beds', 'Number Of Studios Units', 'Number Of Units',
    'One Bedroom Asking Rent/Bed', 'One Bedroom Asking Rent/SF', 'One Bedroom Asking Rent/Unit', 'One Bedroom Avg SF',
    'One Bedroom Concessions %', 'One Bedroom Effective Rent/Bed', 'One Bedroom Effective Rent/SF',
    'One Bedroom Effective Rent/Unit', 'One Bedroom Vacancy %', 'One Bedroom Vacant Units', 'Parking Spaces/Room',
    'Parking Spaces/Unit', 'Proposed Brand', 'Proposed Brand Start Date', 'Studio Asking Rent/Bed',
    'Studio Asking Rent/SF', 'Studio Asking Rent/Unit', 'Studio Avg SF', 'Studio Concessions %',
    'Studio Effective Rent/Bed', 'Studio Effective Rent/SF', 'Studio Effective Rent/Unit', 'Studio Vacancy %',
    'Studio Vacant Units', 'Three Bedroom Asking Rent/Bed', 'Three Bedroom Asking Rent/SF',
    'Three Bedroom Asking Rent/Unit', 'Three Bedroom Avg SF', 'Three Bedroom Concessions %',
    'Three Bedroom Effective Rent/Bed', 'Three Bedroom Effective Rent/SF', 'Three Bedroom Effective Rent/Unit',
    'Three Bedroom Vacancy %', 'Three Bedroom Vacant Units', 'Two Bedroom Asking Rent/Bed',
    'Two Bedroom Asking Rent/SF', 'Two Bedroom Asking Rent/Unit', 'Two Bedroom Avg SF', 'Two Bedroom Concessions %',
    'Two Bedroom Effective Rent/Bed', 'Two Bedroom Effective Rent/SF', 'Two Bedroom Effective Rent/Unit',
    'Two Bedroom Vacancy %', 'Two Bedroom Vacant Units',

]

high_null_count_columns = [
    'Property Name', 'Energy Star', 'LEED Certified', '$Price/Unit', '% 1-Bed', '% 2-Bed', '% 3-Bed', '% 4-Bed',
    '% Studios', 'Affordable Type', 'Anchor GLA', 'Anchor Tenants', 'Architect Name', 'Avg Asking/Bed', 'Avg Asking/SF',
    'Avg Asking/Unit', 'Avg Concessions %', 'Avg Effective/SF', 'Avg Effective/Unit', 'Avg Rent-Direct (Retail)',
    'Avg Rent-Sublet (Industrial)', 'Avg Rent-Sublet (Office)', 'Avg Rent-Sublet (Retail)', 'Avg Unit SF', 'Brand',
    'Closest Transit Stop', 'Closest Transit Stop Dist (mi)', 'Closest Transit Stop Walk Time (min)', 'Core Factor',
    'Data Participation', 'Exp Year', 'Expansion Rooms', 'Expansion Status', 'Expansion Status Month',
    'Expansion Status Year', 'For Sale Price/Room', 'Four Bedroom Asking Rent/Bed', 'Four Bedroom Asking Rent/SF',
    'Four Bedroom Asking Rent/Unit', 'Four Bedroom Avg SF', 'Four Bedroom Concessions %',
    'Four Bedroom Effective Rent/Bed', 'Four Bedroom Effective Rent/SF', 'Four Bedroom Effective Rent/Unit',
    'Four Bedroom Vacancy %', 'Four Bedroom Vacant Units', 'Hotel Class', 'Hotel Grade', 'Hotel Location Type',
    'Hotel Open Date', 'Hotel Operator', 'Interest Rate', 'Interest Rate Type', 'Lab Space (SF)',
    'Lab Space Percent Composition', 'Leasing Company Address', 'Leasing Company City State Zip', 'Leasing Company Fax',
    'Loan Type', 'Market Segment', 'Maturity Date', 'Max Contig Mtg Space', 'Month Built', 'Month Renovated',
    'Mtg Rooms', 'Number Of 1 Bedrooms Units', 'Number Of 2 Bedrooms Units', 'Number Of 3 Bedrooms Units',
    'Number Of 4 Bedrooms Units', 'Number of Beds', 'Number Of Studios Units', 'Number Of Units',
    'One Bedroom Asking Rent/Bed', 'One Bedroom Asking Rent/SF', 'One Bedroom Asking Rent/Unit', 'One Bedroom Avg SF',
    'One Bedroom Concessions %', 'One Bedroom Effective Rent/Bed', 'One Bedroom Effective Rent/SF',
    'One Bedroom Effective Rent/Unit', 'One Bedroom Vacancy %', 'One Bedroom Vacant Units', 'Operation Type',
    'Operational Status', 'Ops Expense', 'Ops Expense Per SF', 'Parent Company', 'Parking Spaces/Room',
    'Parking Spaces/Unit', 'Pre-Leasing', 'Primary Corridors', 'Property Location', 'Proposed Brand',
    'Proposed Brand Start Date', 'Proposed Land Use', 'Recorded Owner Contact', 'Rent Type', 'Rooms',
    'Studio Asking Rent/Bed', 'Studio Asking Rent/SF', 'Studio Asking Rent/Unit', 'Studio Avg SF',
    'Studio Concessions %', 'Studio Effective Rent/Bed', 'Studio Effective Rent/SF', 'Studio Effective Rent/Unit',
    'Studio Vacancy %', 'Studio Vacant Units', 'Style', 'Sublet Services', 'Total Buildings',
    'Three Bedroom Asking Rent/Bed', 'Three Bedroom Asking Rent/SF', 'Three Bedroom Asking Rent/Unit',
    'Three Bedroom Avg SF', 'Three Bedroom Concessions %', 'Three Bedroom Effective Rent/Bed',
    'Three Bedroom Effective Rent/SF', 'Three Bedroom Effective Rent/Unit', 'Three Bedroom Vacancy %',
    'Three Bedroom Vacant Units', 'Total Mtg Space', 'Total New Space (SF)', 'Two Bedroom Asking Rent/Unit',
    'Two Bedroom Avg SF', 'Two Bedroom Concessions %', 'Two Bedroom Effective Rent/Bed',
    'Two Bedroom Effective Rent/SF', 'Two Bedroom Effective Rent/Unit', 'Two Bedroom Vacancy %',
    'Two Bedroom Vacant Units'
]

# categorical columns where 99% of rows are all the same value
categorical_majority_value_columns = [
    'Property Type', 'Scale'
]


not_sure_columns = [
    'Column Spacing', 'Power', 'Amenities', 'Cap Rate', 'Features', 'University'
]


#  -------------
# Final cleaning
# --------------
ignored_columns = (
    not_relevant_columns + not_relevant_residential_columns + high_null_count_columns +
    not_sure_columns + categorical_majority_value_columns
)

location_columns = [col for col in location_columns if col not in ignored_columns]
categorical_columns = [col for col in categorical_columns if col not in ignored_columns]
numerical_columns = [col for col in numerical_columns if col not in ignored_columns]
