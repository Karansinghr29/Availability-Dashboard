"""
india_locations.py
==================

Geographic reference data for India city / state lookup and fuzzy search.

This is NOT business inventory data. It is a reusable India gazetteer used by:
  * city → state resolution (roommate_matching.resolve_state)
  * dashboard autocomplete (Customer City / State)

No apartment / room / rent / occupancy values live here.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Tuple

from utils import normalize_text

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def location_key(value) -> Optional[str]:
    """Case / space / punctuation-insensitive key."""
    norm = normalize_text(value)
    if not norm:
        return None
    return " ".join(_PUNCT_RE.sub(" ", norm).split())


# Canonical Indian states / UTs (display names).
INDIA_STATES: Tuple[str, ...] = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
)

# Common aliases → canonical state name.
_STATE_ALIASES = {
    "tn": "Tamil Nadu", "tamilnadu": "Tamil Nadu", "tamil nadu": "Tamil Nadu",
    "kl": "Kerala", "ka": "Karnataka", "kar": "Karnataka",
    "ts": "Telangana", "tg": "Telangana", "ap": "Andhra Pradesh",
    "mh": "Maharashtra", "maha": "Maharashtra",
    "up": "Uttar Pradesh", "uttarpradesh": "Uttar Pradesh",
    "wb": "West Bengal", "orissa": "Odisha", "odisha": "Odisha",
    "nct of delhi": "Delhi", "new delhi": "Delhi", "ncr": "Delhi",
    "pondicherry": "Puducherry", "pon": "Puducherry",
    "jammu & kashmir": "Jammu and Kashmir", "j&k": "Jammu and Kashmir",
}

# Additional state codes / aliases / common variants (production coverage).
_STATE_ALIASES.update(
    {
        "cg": "Chhattisgarh", "chattisgarh": "Chhattisgarh", "chhatisgarh": "Chhattisgarh",
        "mp": "Madhya Pradesh", "rj": "Rajasthan", "raj": "Rajasthan",
        "gj": "Gujarat", "guj": "Gujarat",
        "br": "Bihar", "jh": "Jharkhand", "od": "Odisha",
        "as": "Assam", "ga": "Goa",
        "dl": "Delhi", "delhi ncr": "Delhi", "national capital territory": "Delhi",
        "hr": "Haryana", "pb": "Punjab", "ch": "Chandigarh",
        "hp": "Himachal Pradesh", "uttaranchal": "Uttarakhand", "ua": "Uttarakhand",
        "pondy": "Puducherry", "puducherry": "Puducherry",
        "jk": "Jammu and Kashmir", "j and k": "Jammu and Kashmir",
        "andaman": "Andaman and Nicobar Islands", "a&n": "Andaman and Nicobar Islands",
        "bengal": "West Bengal", "w bengal": "West Bengal", "west bengal": "West Bengal",
        "tn": "Tamil Nadu",
    }
)

# Major / common Indian cities & towns → state (geographic reference).
# Includes TN district towns requested for autocomplete (Sivagangai, Hosur, etc.).
_INDIA_CITY_STATE_RAW = {
    # Tamil Nadu
    "Chennai": "Tamil Nadu", "Madras": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Madurai": "Tamil Nadu",
    "Tiruchirappalli": "Tamil Nadu", "Trichy": "Tamil Nadu", "Tiruchi": "Tamil Nadu",
    "Salem": "Tamil Nadu", "Erode": "Tamil Nadu", "Tirunelveli": "Tamil Nadu",
    "Vellore": "Tamil Nadu", "Thoothukudi": "Tamil Nadu", "Tuticorin": "Tamil Nadu",
    "Tiruppur": "Tamil Nadu", "Tirupur": "Tamil Nadu", "Thanjavur": "Tamil Nadu",
    "Dindigul": "Tamil Nadu", "Nagercoil": "Tamil Nadu", "Kanyakumari": "Tamil Nadu",
    "Sivagangai": "Tamil Nadu", "Sivaganga": "Tamil Nadu",
    "Hosur": "Tamil Nadu", "Karur": "Tamil Nadu", "Namakkal": "Tamil Nadu",
    "Cuddalore": "Tamil Nadu", "Kanchipuram": "Tamil Nadu", "Kancheepuram": "Tamil Nadu",
    "Chengalpattu": "Tamil Nadu", "Villupuram": "Tamil Nadu", "Viluppuram": "Tamil Nadu",
    "Tiruvannamalai": "Tamil Nadu", "Thiruvannamalai": "Tamil Nadu",
    "Ranipet": "Tamil Nadu", "Tirupattur": "Tamil Nadu", "Krishnagiri": "Tamil Nadu",
    "Dharmapuri": "Tamil Nadu", "Ariyalur": "Tamil Nadu", "Perambalur": "Tamil Nadu",
    "Pudukkottai": "Tamil Nadu", "Sivakasi": "Tamil Nadu", "Virudhunagar": "Tamil Nadu",
    "Ramanathapuram": "Tamil Nadu", "Theni": "Tamil Nadu", "Nilgiris": "Tamil Nadu",
    "Udhagamandalam": "Tamil Nadu", "Ooty": "Tamil Nadu", "Coonoor": "Tamil Nadu",
    "Nagapattinam": "Tamil Nadu", "Mayiladuthurai": "Tamil Nadu", "Karaikudi": "Tamil Nadu",
    "Pollachi": "Tamil Nadu", "Udumalaipettai": "Tamil Nadu", "Gobi": "Tamil Nadu",
    "Gobichettipalayam": "Tamil Nadu", "Mettupalayam": "Tamil Nadu",
    "Tambaram": "Tamil Nadu", "Avadi": "Tamil Nadu", "Ambattur": "Tamil Nadu",
    "Tiruvallur": "Tamil Nadu", "Thiruvallur": "Tamil Nadu",
    "Rajapalayam": "Tamil Nadu", "Tenkasi": "Tamil Nadu", "Ambur": "Tamil Nadu",
    "Vaniyambadi": "Tamil Nadu", "Tindivanam": "Tamil Nadu", "Chidambaram": "Tamil Nadu",
    # Kerala
    "Kochi": "Kerala", "Cochin": "Kerala", "Ernakulam": "Kerala",
    "Thiruvananthapuram": "Kerala", "Trivandrum": "Kerala",
    "Kozhikode": "Kerala", "Calicut": "Kerala", "Thrissur": "Kerala", "Trichur": "Kerala",
    "Kollam": "Kerala", "Kannur": "Kerala", "Cannanore": "Kerala",
    "Kottayam": "Kerala", "Palakkad": "Kerala", "Palghat": "Kerala",
    "Alappuzha": "Kerala", "Alleppey": "Kerala", "Malappuram": "Kerala",
    "Pathanamthitta": "Kerala", "Idukki": "Kerala", "Wayanad": "Kerala",
    "Kasaragod": "Kerala", "Guruvayur": "Kerala",
    # Karnataka
    "Bengaluru": "Karnataka", "Bangalore": "Karnataka", "Mysuru": "Karnataka",
    "Mysore": "Karnataka", "Mangaluru": "Karnataka", "Mangalore": "Karnataka",
    "Hubli": "Karnataka", "Hubballi": "Karnataka", "Dharwad": "Karnataka",
    "Belagavi": "Karnataka", "Belgaum": "Karnataka", "Davangere": "Karnataka",
    "Shivamogga": "Karnataka", "Shimoga": "Karnataka", "Tumakuru": "Karnataka",
    "Tumkur": "Karnataka", "Ballari": "Karnataka", "Bellary": "Karnataka",
    "Kalaburagi": "Karnataka", "Gulbarga": "Karnataka", "Vijayapura": "Karnataka",
    "Bijapur": "Karnataka", "Udupi": "Karnataka", "Hassan": "Karnataka",
    "Mandya": "Karnataka", "Chikkamagaluru": "Karnataka", "Raichur": "Karnataka",
    # Telangana / Andhra
    "Hyderabad": "Telangana", "Secunderabad": "Telangana", "Warangal": "Telangana",
    "Nizamabad": "Telangana", "Karimnagar": "Telangana", "Khammam": "Telangana",
    "Mahbubnagar": "Telangana", "Nalgonda": "Telangana", "Adilabad": "Telangana",
    "Visakhapatnam": "Andhra Pradesh", "Vizag": "Andhra Pradesh",
    "Vijayawada": "Andhra Pradesh", "Guntur": "Andhra Pradesh", "Nellore": "Andhra Pradesh",
    "Tirupati": "Andhra Pradesh", "Kurnool": "Andhra Pradesh",
    "Rajahmundry": "Andhra Pradesh", "Kakinada": "Andhra Pradesh",
    "Anantapur": "Andhra Pradesh", "Kadapa": "Andhra Pradesh", "Cuddapah": "Andhra Pradesh",
    "Ongole": "Andhra Pradesh", "Eluru": "Andhra Pradesh", "Srikakulam": "Andhra Pradesh",
    # Maharashtra
    "Mumbai": "Maharashtra", "Bombay": "Maharashtra", "Pune": "Maharashtra",
    "Nagpur": "Maharashtra", "Nashik": "Maharashtra", "Nasik": "Maharashtra",
    "Aurangabad": "Maharashtra", "Chhatrapati Sambhajinagar": "Maharashtra",
    "Solapur": "Maharashtra", "Thane": "Maharashtra", "Navi Mumbai": "Maharashtra",
    "Kolhapur": "Maharashtra", "Amravati": "Maharashtra", "Nanded": "Maharashtra",
    "Sangli": "Maharashtra", "Satara": "Maharashtra", "Jalna": "Maharashtra",
    "Akola": "Maharashtra", "Latur": "Maharashtra", "Ahmednagar": "Maharashtra",
    "Sindhudurg": "Maharashtra", "Ratnagiri": "Maharashtra", "Palghar": "Maharashtra",
    "Kalyan": "Maharashtra", "Vasai": "Maharashtra", "Panvel": "Maharashtra",
    # Delhi NCR / Haryana / UP
    "Delhi": "Delhi", "New Delhi": "Delhi",
    "Gurgaon": "Haryana", "Gurugram": "Haryana", "Faridabad": "Haryana",
    "Panipat": "Haryana", "Ambala": "Haryana", "Karnal": "Haryana", "Hisar": "Haryana",
    "Rohtak": "Haryana", "Sonipat": "Haryana", "Yamunanagar": "Haryana",
    "Noida": "Uttar Pradesh", "Greater Noida": "Uttar Pradesh",
    "Ghaziabad": "Uttar Pradesh", "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
    "Agra": "Uttar Pradesh", "Varanasi": "Uttar Pradesh", "Banaras": "Uttar Pradesh",
    "Prayagraj": "Uttar Pradesh", "Allahabad": "Uttar Pradesh", "Meerut": "Uttar Pradesh",
    "Bareilly": "Uttar Pradesh", "Aligarh": "Uttar Pradesh", "Gorakhpur": "Uttar Pradesh",
    "Moradabad": "Uttar Pradesh", "Saharanpur": "Uttar Pradesh", "Mathura": "Uttar Pradesh",
    "Jhansi": "Uttar Pradesh", "Firozabad": "Uttar Pradesh",
    # West Bengal / East
    "Kolkata": "West Bengal", "Calcutta": "West Bengal", "Howrah": "West Bengal",
    "Durgapur": "West Bengal", "Asansol": "West Bengal", "Siliguri": "West Bengal",
    "Kharagpur": "West Bengal", "Darjeeling": "West Bengal",
    "Patna": "Bihar", "Gaya": "Bihar", "Bhagalpur": "Bihar", "Muzaffarpur": "Bihar",
    "Purnia": "Bihar", "Darbhanga": "Bihar",
    "Ranchi": "Jharkhand", "Jamshedpur": "Jharkhand", "Dhanbad": "Jharkhand",
    "Bokaro": "Jharkhand", "Deoghar": "Jharkhand",
    "Bhubaneswar": "Odisha", "Cuttack": "Odisha", "Rourkela": "Odisha",
    "Puri": "Odisha", "Sambalpur": "Odisha", "Berhampur": "Odisha",
    # Gujarat / Rajasthan / MP
    "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Vadodara": "Gujarat", "Baroda": "Gujarat",
    "Rajkot": "Gujarat", "Bhavnagar": "Gujarat", "Gandhinagar": "Gujarat",
    "Jamnagar": "Gujarat", "Junagadh": "Gujarat", "Anand": "Gujarat",
    "Jaipur": "Rajasthan", "Jodhpur": "Rajasthan", "Udaipur": "Rajasthan",
    "Kota": "Rajasthan", "Ajmer": "Rajasthan", "Bikaner": "Rajasthan",
    "Sikar": "Rajasthan", "Alwar": "Rajasthan", "Bhilwara": "Rajasthan",
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh", "Jabalpur": "Madhya Pradesh",
    "Gwalior": "Madhya Pradesh", "Ujjain": "Madhya Pradesh", "Sagar": "Madhya Pradesh",
    "Rewa": "Madhya Pradesh", "Satna": "Madhya Pradesh",
    "Raipur": "Chhattisgarh", "Bhilai": "Chhattisgarh", "Bilaspur": "Chhattisgarh",
    "Durg": "Chhattisgarh", "Korba": "Chhattisgarh",
    # North / NE / Goa
    "Ludhiana": "Punjab", "Amritsar": "Punjab", "Jalandhar": "Punjab", "Patiala": "Punjab",
    "Mohali": "Punjab", "Bathinda": "Punjab",
    "Chandigarh": "Chandigarh",
    "Srinagar": "Jammu and Kashmir", "Jammu": "Jammu and Kashmir",
    "Leh": "Ladakh", "Kargil": "Ladakh",
    "Shimla": "Himachal Pradesh", "Dharamshala": "Himachal Pradesh", "Manali": "Himachal Pradesh",
    "Dehradun": "Uttarakhand", "Haridwar": "Uttarakhand", "Rishikesh": "Uttarakhand",
    "Haldwani": "Uttarakhand", "Nainital": "Uttarakhand",
    "Guwahati": "Assam", "Dispur": "Assam", "Silchar": "Assam", "Dibrugarh": "Assam",
    "Jorhat": "Assam", "Tezpur": "Assam",
    "Imphal": "Manipur", "Shillong": "Meghalaya", "Agartala": "Tripura",
    "Aizawl": "Mizoram", "Kohima": "Nagaland", "Dimapur": "Nagaland",
    "Itanagar": "Arunachal Pradesh", "Gangtok": "Sikkim",
    "Panaji": "Goa", "Panjim": "Goa", "Margao": "Goa", "Vasco": "Goa", "Mapusa": "Goa",
    "Puducherry": "Puducherry", "Pondicherry": "Puducherry", "Oulgaret": "Puducherry",
    "Port Blair": "Andaman and Nicobar Islands",
    "Kavaratti": "Lakshadweep",
    "Silvassa": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman": "Dadra and Nagar Haveli and Daman and Diu",
    "Diu": "Dadra and Nagar Haveli and Daman and Diu",
}

# --------------------------------------------------------------------------- #
# Comprehensive all-India coverage (district HQs, major towns, aliases,
# common alternate spellings). Merged onto the seed map above so existing
# entries are preserved. Purely geographic reference data — no business values.
# For genuinely ambiguous names shared by two states, the more populous /
# widely-known location is chosen.
# --------------------------------------------------------------------------- #
_INDIA_CITY_STATE_RAW.update(
    {
        # ---------------- Tamil Nadu (district towns + requested) ----------------
        "Devakottai": "Tamil Nadu", "Devekottai": "Tamil Nadu",
        "Aruppukkottai": "Tamil Nadu", "Paramakudi": "Tamil Nadu",
        "Manamadurai": "Tamil Nadu", "Tiruchendur": "Tamil Nadu",
        "Kovilpatti": "Tamil Nadu", "Srivilliputhur": "Tamil Nadu",
        "Arakkonam": "Tamil Nadu", "Gudiyatham": "Tamil Nadu",
        "Palani": "Tamil Nadu", "Oddanchatram": "Tamil Nadu",
        "Kumbakonam": "Tamil Nadu", "Mannargudi": "Tamil Nadu",
        "Pattukkottai": "Tamil Nadu", "Arani": "Tamil Nadu",
        "Marthandam": "Tamil Nadu", "Sankarankovil": "Tamil Nadu",
        "Sattur": "Tamil Nadu", "Melur": "Tamil Nadu", "Usilampatti": "Tamil Nadu",
        "Bodinayakanur": "Tamil Nadu", "Cumbum": "Tamil Nadu",
        "Palladam": "Tamil Nadu", "Dharapuram": "Tamil Nadu", "Kangayam": "Tamil Nadu",
        "Vandavasi": "Tamil Nadu", "Gingee": "Tamil Nadu", "Ulundurpet": "Tamil Nadu",
        "Perundurai": "Tamil Nadu", "Bhavani": "Tamil Nadu", "Sathyamangalam": "Tamil Nadu",
        "Rasipuram": "Tamil Nadu", "Tiruchengode": "Tamil Nadu", "Komarapalayam": "Tamil Nadu",
        "Aranthangi": "Tamil Nadu", "Thiruvarur": "Tamil Nadu", "Kovilpattai": "Tamil Nadu",
        # ---------------- Kerala ----------------
        "Manjeri": "Kerala", "Tirur": "Kerala", "Ponnani": "Kerala", "Perinthalmanna": "Kerala",
        "Ottapalam": "Kerala", "Chalakudy": "Kerala", "Kodungallur": "Kerala",
        "Aluva": "Kerala", "Perumbavoor": "Kerala", "Muvattupuzha": "Kerala",
        "Thodupuzha": "Kerala", "Pala": "Kerala", "Changanassery": "Kerala",
        "Thiruvalla": "Kerala", "Chengannur": "Kerala", "Cherthala": "Kerala",
        "Mavelikkara": "Kerala", "Kayamkulam": "Kerala", "Karunagappally": "Kerala",
        "Nedumangad": "Kerala", "Neyyattinkara": "Kerala", "Attingal": "Kerala",
        "Vatakara": "Kerala", "Koyilandy": "Kerala", "Payyanur": "Kerala",
        "Taliparamba": "Kerala", "Thalassery": "Kerala", "Tellicherry": "Kerala",
        "Nilambur": "Kerala", "Kalpetta": "Kerala",
        # ---------------- Karnataka ----------------
        "Chitradurga": "Karnataka", "Bidar": "Karnataka", "Bagalkot": "Karnataka",
        "Gadag": "Karnataka", "Haveri": "Karnataka", "Koppal": "Karnataka",
        "Yadgir": "Karnataka", "Chikkaballapur": "Karnataka", "Kolar": "Karnataka",
        "Chamarajanagar": "Karnataka", "Kodagu": "Karnataka", "Madikeri": "Karnataka",
        "Karwar": "Karnataka", "Sirsi": "Karnataka", "Bhadravati": "Karnataka",
        "Robertsonpet": "Karnataka", "Gangavati": "Karnataka", "Ranebennuru": "Karnataka",
        "Hospet": "Karnataka", "Hosapete": "Karnataka", "Chikmagalur": "Karnataka",
        "Ramanagara": "Karnataka", "Doddaballapur": "Karnataka",
        # ---------------- Telangana ----------------
        "Ramagundam": "Telangana", "Siddipet": "Telangana", "Miryalaguda": "Telangana",
        "Suryapet": "Telangana", "Jagtial": "Telangana", "Mancherial": "Telangana",
        "Nirmal": "Telangana", "Kamareddy": "Telangana", "Sangareddy": "Telangana",
        "Medak": "Telangana", "Wanaparthy": "Telangana", "Gadwal": "Telangana",
        "Bhongir": "Telangana", "Zaheerabad": "Telangana",
        # ---------------- Andhra Pradesh ----------------
        "Rajamahendravaram": "Andhra Pradesh", "Machilipatnam": "Andhra Pradesh",
        "Tenali": "Andhra Pradesh", "Chittoor": "Andhra Pradesh", "Hindupur": "Andhra Pradesh",
        "Proddatur": "Andhra Pradesh", "Adoni": "Andhra Pradesh", "Nandyal": "Andhra Pradesh",
        "Madanapalle": "Andhra Pradesh", "Tadepalligudem": "Andhra Pradesh",
        "Bhimavaram": "Andhra Pradesh", "Narasaraopet": "Andhra Pradesh",
        "Chirala": "Andhra Pradesh", "Gudivada": "Andhra Pradesh", "Vizianagaram": "Andhra Pradesh",
        "Tirupathi": "Andhra Pradesh", "Amaravati": "Andhra Pradesh", "Vishakhapatnam": "Andhra Pradesh",
        "Anakapalle": "Andhra Pradesh", "Dharmavaram": "Andhra Pradesh",
        # ---------------- Maharashtra ----------------
        "Pune": "Maharashtra", "Poona": "Maharashtra", "Pimpri": "Maharashtra",
        "Pimpri Chinchwad": "Maharashtra", "Chinchwad": "Maharashtra", "Ichalkaranji": "Maharashtra",
        "Jalgaon": "Maharashtra", "Dhule": "Maharashtra", "Nandurbar": "Maharashtra",
        "Beed": "Maharashtra", "Osmanabad": "Maharashtra", "Dharashiv": "Maharashtra",
        "Parbhani": "Maharashtra", "Hingoli": "Maharashtra", "Washim": "Maharashtra",
        "Yavatmal": "Maharashtra", "Wardha": "Maharashtra", "Chandrapur": "Maharashtra",
        "Gondia": "Maharashtra", "Bhandara": "Maharashtra", "Gadchiroli": "Maharashtra",
        "Buldhana": "Maharashtra", "Malegaon": "Maharashtra", "Bhusawal": "Maharashtra",
        "Miraj": "Maharashtra", "Barshi": "Maharashtra", "Baramati": "Maharashtra",
        "Chhatrapati Sambhaji Nagar": "Maharashtra", "Mira Bhayandar": "Maharashtra",
        "Ulhasnagar": "Maharashtra", "Ambernath": "Maharashtra", "Badlapur": "Maharashtra",
        "Bhiwandi": "Maharashtra", "Karad": "Maharashtra",
        # ---------------- Gujarat ----------------
        "Nadiad": "Gujarat", "Mehsana": "Gujarat", "Morbi": "Gujarat", "Surendranagar": "Gujarat",
        "Bharuch": "Gujarat", "Navsari": "Gujarat", "Valsad": "Gujarat", "Vapi": "Gujarat",
        "Porbandar": "Gujarat", "Veraval": "Gujarat", "Godhra": "Gujarat", "Palanpur": "Gujarat",
        "Patan": "Gujarat", "Botad": "Gujarat", "Amreli": "Gujarat", "Bhuj": "Gujarat",
        "Gandhidham": "Gujarat", "Ankleshwar": "Gujarat", "Dahod": "Gujarat", "Deesa": "Gujarat",
        # ---------------- Rajasthan ----------------
        "Bharatpur": "Rajasthan", "Pali": "Rajasthan", "Sri Ganganagar": "Rajasthan",
        "Hanumangarh": "Rajasthan", "Churu": "Rajasthan", "Jhunjhunu": "Rajasthan",
        "Nagaur": "Rajasthan", "Tonk": "Rajasthan", "Banswara": "Rajasthan",
        "Chittorgarh": "Rajasthan", "Dungarpur": "Rajasthan", "Barmer": "Rajasthan",
        "Jaisalmer": "Rajasthan", "Dausa": "Rajasthan", "Sawai Madhopur": "Rajasthan",
        "Beawar": "Rajasthan", "Kishangarh": "Rajasthan", "Bundi": "Rajasthan",
        # ---------------- Madhya Pradesh ----------------
        "Ratlam": "Madhya Pradesh", "Dewas": "Madhya Pradesh", "Burhanpur": "Madhya Pradesh",
        "Khandwa": "Madhya Pradesh", "Chhindwara": "Madhya Pradesh", "Katni": "Madhya Pradesh",
        "Singrauli": "Madhya Pradesh", "Vidisha": "Madhya Pradesh", "Guna": "Madhya Pradesh",
        "Shivpuri": "Madhya Pradesh", "Mandsaur": "Madhya Pradesh", "Neemuch": "Madhya Pradesh",
        "Chhatarpur": "Madhya Pradesh", "Damoh": "Madhya Pradesh", "Hoshangabad": "Madhya Pradesh",
        "Narmadapuram": "Madhya Pradesh", "Balaghat": "Madhya Pradesh", "Seoni": "Madhya Pradesh",
        "Morena": "Madhya Pradesh", "Bhind": "Madhya Pradesh",
        # ---------------- Chhattisgarh ----------------
        "Raigarh": "Chhattisgarh", "Jagdalpur": "Chhattisgarh", "Ambikapur": "Chhattisgarh",
        "Rajnandgaon": "Chhattisgarh", "Dhamtari": "Chhattisgarh", "Mahasamund": "Chhattisgarh",
        "Bhilai Nagar": "Chhattisgarh", "Kanker": "Chhattisgarh",
        # ---------------- Uttar Pradesh ----------------
        "Jaunpur": "Uttar Pradesh", "Rae Bareli": "Uttar Pradesh", "Sultanpur": "Uttar Pradesh",
        "Faizabad": "Uttar Pradesh", "Ayodhya": "Uttar Pradesh", "Basti": "Uttar Pradesh",
        "Azamgarh": "Uttar Pradesh", "Mau": "Uttar Pradesh", "Ballia": "Uttar Pradesh",
        "Ghazipur": "Uttar Pradesh", "Mirzapur": "Uttar Pradesh", "Deoria": "Uttar Pradesh",
        "Sitapur": "Uttar Pradesh", "Hardoi": "Uttar Pradesh", "Unnao": "Uttar Pradesh",
        "Etawah": "Uttar Pradesh", "Mainpuri": "Uttar Pradesh", "Etah": "Uttar Pradesh",
        "Budaun": "Uttar Pradesh", "Shahjahanpur": "Uttar Pradesh", "Rampur": "Uttar Pradesh",
        "Amroha": "Uttar Pradesh", "Sambhal": "Uttar Pradesh", "Bijnor": "Uttar Pradesh",
        "Muzaffarnagar": "Uttar Pradesh", "Shamli": "Uttar Pradesh", "Bulandshahr": "Uttar Pradesh",
        "Hapur": "Uttar Pradesh", "Hathras": "Uttar Pradesh", "Kashi": "Uttar Pradesh",
        "Fatehpur": "Uttar Pradesh", "Pratapgarh": "Uttar Pradesh", "Gonda": "Uttar Pradesh",
        "Bahraich": "Uttar Pradesh", "Barabanki": "Uttar Pradesh", "Lakhimpur": "Uttar Pradesh",
        # ---------------- Bihar ----------------
        "Begusarai": "Bihar", "Chapra": "Bihar", "Katihar": "Bihar", "Munger": "Bihar",
        "Chhapra": "Bihar", "Arrah": "Bihar", "Ara": "Bihar", "Buxar": "Bihar",
        "Sasaram": "Bihar", "Bettiah": "Bihar", "Motihari": "Bihar", "Siwan": "Bihar",
        "Hajipur": "Bihar", "Sitamarhi": "Bihar", "Madhubani": "Bihar", "Samastipur": "Bihar",
        "Nawada": "Bihar", "Jamui": "Bihar", "Kishanganj": "Bihar", "Araria": "Bihar",
        "Saharsa": "Bihar", "Bhabua": "Bihar", "Aurangabad Bihar": "Bihar",
        # ---------------- Jharkhand ----------------
        "Hazaribagh": "Jharkhand", "Giridih": "Jharkhand", "Ramgarh": "Jharkhand",
        "Medininagar": "Jharkhand", "Daltonganj": "Jharkhand", "Chaibasa": "Jharkhand",
        "Dumka": "Jharkhand", "Phusro": "Jharkhand", "Chirkunda": "Jharkhand",
        # ---------------- Odisha ----------------
        "Balasore": "Odisha", "Baleswar": "Odisha", "Baripada": "Odisha", "Bhadrak": "Odisha",
        "Jharsuguda": "Odisha", "Jeypore": "Odisha", "Angul": "Odisha", "Dhenkanal": "Odisha",
        "Kendujhar": "Odisha", "Keonjhar": "Odisha", "Rayagada": "Odisha", "Koraput": "Odisha",
        "Balangir": "Odisha", "Bargarh": "Odisha", "Paradip": "Odisha",
        # ---------------- West Bengal ----------------
        "Malda": "West Bengal", "English Bazar": "West Bengal", "Berhampore": "West Bengal",
        "Baharampur": "West Bengal", "Krishnanagar": "West Bengal", "Bardhaman": "West Bengal",
        "Burdwan": "West Bengal", "Bankura": "West Bengal", "Purulia": "West Bengal",
        "Raiganj": "West Bengal", "Jalpaiguri": "West Bengal", "Cooch Behar": "West Bengal",
        "Alipurduar": "West Bengal", "Haldia": "West Bengal", "Midnapore": "West Bengal",
        "Barasat": "West Bengal", "Bidhannagar": "West Bengal", "Chandannagar": "West Bengal",
        "Serampore": "West Bengal", "Habra": "West Bengal", "Ranaghat": "West Bengal",
        # ---------------- Punjab ----------------
        "Hoshiarpur": "Punjab", "Moga": "Punjab", "Firozpur": "Punjab", "Ferozepur": "Punjab",
        "Pathankot": "Punjab", "Kapurthala": "Punjab", "Phagwara": "Punjab", "Batala": "Punjab",
        "Barnala": "Punjab", "Muktsar": "Punjab", "Malerkotla": "Punjab", "Khanna": "Punjab",
        "Rajpura": "Punjab", "Sangrur": "Punjab", "Zirakpur": "Punjab", "Fazilka": "Punjab",
        # ---------------- Haryana ----------------
        "Bhiwani": "Haryana", "Bahadurgarh": "Haryana", "Jind": "Haryana", "Kaithal": "Haryana",
        "Rewari": "Haryana", "Palwal": "Haryana", "Kurukshetra": "Haryana", "Sirsa": "Haryana",
        "Fatehabad": "Haryana", "Jhajjar": "Haryana", "Narnaul": "Haryana", "Gohana": "Haryana",
        "Thanesar": "Haryana", "Charkhi Dadri": "Haryana",
        # ---------------- Uttarakhand ----------------
        "Roorkee": "Uttarakhand", "Kashipur": "Uttarakhand", "Rudrapur": "Uttarakhand",
        "Kotdwar": "Uttarakhand", "Pithoragarh": "Uttarakhand", "Almora": "Uttarakhand",
        "Mussoorie": "Uttarakhand", "Ramnagar": "Uttarakhand",
        # ---------------- Himachal Pradesh ----------------
        "Solan": "Himachal Pradesh", "Mandi": "Himachal Pradesh", "Kullu": "Himachal Pradesh",
        "Palampur": "Himachal Pradesh", "Nahan": "Himachal Pradesh", "Una": "Himachal Pradesh",
        "Hamirpur HP": "Himachal Pradesh", "Chamba": "Himachal Pradesh", "Baddi": "Himachal Pradesh",
        # ---------------- Jammu & Kashmir / Ladakh ----------------
        "Anantnag": "Jammu and Kashmir", "Baramulla": "Jammu and Kashmir",
        "Sopore": "Jammu and Kashmir", "Udhampur": "Jammu and Kashmir", "Kathua": "Jammu and Kashmir",
        # ---------------- Assam ----------------
        "Nagaon": "Assam", "Tinsukia": "Assam", "Bongaigaon": "Assam", "Diphu": "Assam",
        "Goalpara": "Assam", "Barpeta": "Assam", "North Lakhimpur": "Assam", "Sivasagar": "Assam",
        "Karimganj": "Assam", "Hailakandi": "Assam", "Nalbari": "Assam", "Dhubri": "Assam",
        # ---------------- North-East (other) ----------------
        "Tura": "Meghalaya", "Jowai": "Meghalaya", "Nongstoin": "Meghalaya",
        "Ukhrul": "Manipur", "Thoubal": "Manipur", "Bishnupur": "Manipur",
        "Churachandpur": "Manipur", "Tamenglong": "Manipur",
        "Lunglei": "Mizoram", "Champhai": "Mizoram",
        "Wokha": "Nagaland", "Mokokchung": "Nagaland", "Tuensang": "Nagaland",
        "Naharlagun": "Arunachal Pradesh", "Pasighat": "Arunachal Pradesh", "Tawang": "Arunachal Pradesh",
        "Namchi": "Sikkim", "Gyalshing": "Sikkim", "Mangan": "Sikkim",
        "Udaipur Tripura": "Tripura", "Dharmanagar": "Tripura", "Ambassa": "Tripura",
        # ---------------- Goa ----------------
        "Ponda": "Goa", "Bicholim": "Goa", "Curchorem": "Goa", "Sanquelim": "Goa",
        "Calangute": "Goa", "Candolim": "Goa",
        # ---------------- Union Territories ----------------
        "Karaikal": "Puducherry", "Mahe": "Puducherry", "Yanam": "Puducherry",
        # ---------------- Common alternate spellings / misspellings ----------------
        "Banglore": "Karnataka", "Bengalore": "Karnataka", "Bengaluru City": "Karnataka",
        "Vishakapatnam": "Andhra Pradesh", "Vizag City": "Andhra Pradesh",
        "Trivandram": "Kerala", "Cochi": "Kerala", "Calicut City": "Kerala",
        "Coimbatur": "Tamil Nadu", "Madhurai": "Tamil Nadu", "Puducherry UT": "Puducherry",
        "Gurugram City": "Haryana", "Noida City": "Uttar Pradesh",
        "New Bombay": "Maharashtra", "Bombay City": "Maharashtra",
    }
)

INDIA_CITY_STATE = {location_key(k): v for k, v in _INDIA_CITY_STATE_RAW.items()}
INDIA_CITY_LABELS = {location_key(k): k for k in _INDIA_CITY_STATE_RAW}
INDIA_STATE_KEYS = {location_key(s): s for s in INDIA_STATES}
for alias, canon in _STATE_ALIASES.items():
    INDIA_STATE_KEYS[location_key(alias)] = canon

# Suggestion catalogue: states + city display labels (title-cased originals).
_SUGGESTION_POOL: List[str] = sorted(
    set(INDIA_STATES) | set(_INDIA_CITY_STATE_RAW.keys()),
    key=lambda s: s.lower(),
)


def resolve_india_state(location: str) -> Optional[str]:
    """Resolve a typed city OR state string to a canonical Indian state.

    Order:
      1. Exact state / alias match
      2. Exact city → state from the India gazetteer
    Returns None when unknown.
    """
    key = location_key(location)
    if not key:
        return None
    if key in INDIA_STATE_KEYS:
        return INDIA_STATE_KEYS[key]
    return INDIA_CITY_STATE.get(key)


def is_india_state(location: str) -> bool:
    key = location_key(location)
    return bool(key and key in INDIA_STATE_KEYS)


def suggest_india_locations(query: str, limit: int = 25) -> List[str]:
    """Fuzzy / prefix suggestions over Indian cities and states.

    Typing ``si`` returns Sivagangai, Siliguri, Sikar, Sindhudurg, etc.
    Empty query returns a short starter list (states + popular cities).
    """
    q = (query or "").strip()
    if not q:
        starters = list(INDIA_STATES[:12]) + [
            "Chennai", "Coimbatore", "Madurai", "Bengaluru", "Hyderabad",
            "Mumbai", "Pune", "Delhi", "Kochi", "Kolkata",
        ]
        return starters[:limit]

    q_key = location_key(q) or q.lower()
    scored: list[tuple[float, str]] = []
    for label in _SUGGESTION_POOL:
        lk = location_key(label) or label.lower()
        if lk.startswith(q_key):
            score = 100.0 - (len(lk) - len(q_key)) * 0.1
        elif q_key in lk:
            score = 70.0 - lk.find(q_key) * 0.5
        else:
            ratio = SequenceMatcher(None, q_key, lk).ratio()
            if ratio < 0.55:
                continue
            score = 40.0 * ratio
        scored.append((score, label))

    scored.sort(key=lambda t: (-t[0], t[1].lower()))
    # Deduplicate while preserving order (e.g. Trichy / Tiruchirappalli both kept).
    out, seen = [], set()
    for _, label in scored:
        k = location_key(label)
        if k in seen:
            continue
        seen.add(k)
        out.append(label)
        if len(out) >= limit:
            break
    return out


def merge_city_maps(*maps: Iterable) -> dict:
    """Merge several city→state dicts; later maps override earlier ones."""
    out = {}
    for m in maps:
        if not m:
            continue
        if isinstance(m, dict):
            for k, v in m.items():
                kk = location_key(k)
                if kk and v:
                    out[kk] = v
    return out
