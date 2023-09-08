import dns.resolver
import socket
import telnetlib

def check_mx_record(domain):
    try:
        # Performing MX record lookup
        mx_records = dns.resolver.resolve(domain, 'MX')
        for rdata in mx_records:
            mx_domain = str(rdata.exchange)
            # Resolving MX domain to IP
            ip = socket.gethostbyname(mx_domain)

            # Trying to connect via Telnet on port 25
            try:
                with telnetlib.Telnet(ip, 25, timeout=5) as tn:
                    greeting = tn.read_until(b"\n", timeout=5).decode('ascii').strip()

                    # Check if greeting starts with "220"
                    if greeting.startswith("220"):
                        print(f"{domain} ({mx_domain}, {ip}): OK")
                    else:
                        print(f"{domain} ({mx_domain}, {ip}): NOT OK")

            except Exception as e:
                print(f"{domain} ({mx_domain}, {ip}): NOT OK")

    except dns.resolver.NoAnswer:
        print(f"{domain}: NOT OK (No MX records found)")
    except dns.resolver.NXDOMAIN:
        print(f"{domain}: NOT OK (Domain does not exist)")
    except Exception as e:
        print(f"{domain}: NOT OK (An error occurred: {e})")

# Extended list of popular international and German email domains
popular_domains = [
    # International
    'gmail.com',
    'yahoo.com',
    'hotmail.com',
    'aol.com',
    'outlook.com',
    'live.com',
    'msn.com',
    'icloud.com',
    'zoho.com',
    'mail.com',
    'protonmail.com',
    'yandex.com',
    'fastmail.com',
    'gmx.com',
    'hushmail.com',
    'inbox.com',
    'rediffmail.com',
    'rocketmail.com',
    'excite.com',
    'sbcglobal.net',
    'ymail.com',
    'verizon.net',
    'comcast.net',
    'me.com',
    'mail.ru',
    'earthlink.net',
    'bellsouth.net',
    'junonetworks.com',
    'att.net',
    'cox.net',
    'optonline.net',
    'juno.com',
    'mac.com',
    'blueyonder.co.uk',
    'ntlworld.com',

    # German
    'web.de',
    'gmx.de',
    't-online.de',
    'freenet.de',
    'posteo.de',
    'mail.de',
    '1und1.de',
    'arcor.de',
    'kabelmail.de',
    'ok.de',
    'vodafone.de',
    'online.de',
    'qip.de',
    'alice.de',
    'kabeldeutschland.de',
    'o2online.de',
    'yahoo.de',
    'hotmail.de',
    'gmx.net',
]

# Checking each domain
for domain in popular_domains:
    check_mx_record(domain)
