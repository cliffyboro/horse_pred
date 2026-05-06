# config.py

# List of UK and Irish courses to filter the raw data
UK_IRE_COURSES = {
    'Aintree', 'Ascot', 'Ayr', 'Bangor-on-Dee', 'Bath', 'Beverley', 'Brighton',
    'Carlisle', 'Cartmel', 'Catterick', 'Chelmsford (AW)', 'Cheltenham', 
    'Chepstow', 'Chester', 'Doncaster', 'Epsom', 'Exeter', 'Ffos Las', 
    'Fontwell', 'Goodwood', 'Hamilton', 'Haydock', 'Hexham', 'Huntingdon',
    'Kelso', 'Kempton (AW)', 'Leicester', 'Lingfield (AW)', 'Ludlow', 
    'Market Rasen', 'Musselburgh', 'Newbury', 'Newcastle', 'Newmarket', 
    'Newton Abbot', 'Nottingham', 'Perth', 'Pontefract', 'Redcar', 'Ripon', 
    'Salisbury', 'Sandown', 'Sedgefield', 'Southwell', 'Southwell (AW)', 
    'Stratford', 'Taunton', 'Thirsk', 'Towcester', 'Uttoxeter', 'Warwick', 
    'Wetherby', 'Wincanton', 'Windsor', 'Wolverhampton (AW)', 'Worcester', 
    'Yarmouth', 'York', 'Newmarket (July)', 'Newmarket (Rowley)',
    # Ireland
    'Ballinrobe (IRE)', 'Bellewstown (IRE)', 'Clonmel (IRE)', 'Cork (IRE)',
    'Curragh (IRE)', 'Down Royal (IRE)', 'Downpatrick (IRE)', 'Dundalk (AW) (IRE)',
    'Fairyhouse (IRE)', 'Galway (IRE)', 'Gowran (IRE)', 'Gowran Park (IRE)',
    'Kilbeggan (IRE)', 'Killarney (IRE)', 'Laytown (IRE)', 'Leopardstown (IRE)',
    'Limerick (IRE)', 'Listowel (IRE)', 'Naas (IRE)', 'Navan (IRE)', 
    'Punchestown (IRE)', 'Roscommon (IRE)', 'Sligo (IRE)', 'Tipperary (IRE)',
    'Tramore (IRE)', 'Wexford (IRE)', 'Thurles (IRE)', 'Plumpton', 'Kempton', 'Lingfield'
}

# Database settings
DB_PATH = 'database/horse_racing.db'
RAW_DATA_PATH = 'database/raceform.csv'