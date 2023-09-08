# MX-Record-Checker
## Description

A Python utility to check the compatibility and health of MX records of email domains with their mail servers. This script performs an MX record lookup, resolves the MX domain to its IP, and tries to establish a Telnet connection on port 25 to verify if the mail server is active.

## Prerequisites

- Python 3.x
- dns.resolver
- socket
- telnetlib

You can install the required packages using pip:
```bash
pip install dnspython
```

## Usage

Clone this repository:
```bash
git clone https://github.com/yourusername/MX-Record-Checker.git
```

Navigate to the directory:
```bash
cd MX-Record-Checker
```

Run the script:
```bash
python check_mx_record.py
```
The script will then iterate through a list of popular email domains and check each domain's MX record, resolving it to an IP and attempting to connect via Telnet on port 25. The results will be printed on the console.

## Customization
You can add or remove domains from the popular_domains list in the script to check MX records for specific email domains.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT (https://choosealicense.com/licenses/mit/)
