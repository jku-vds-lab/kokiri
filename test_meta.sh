#!/bin/sh
curl 'http://localhost:8080/kokiri/cmp_meta/' \
  -H 'Accept: */*' \
  -H 'Accept-Language: de,de-DE;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6' \
  -H 'Cache-Control: max-age=0' \
  -H 'Connection: keep-alive' \
  -H 'Content-Type: application/json' \
  -H 'Origin: http://localhost:8080' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36 Edg/101.0.1210.53' \
  -H 'sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="101", "Microsoft Edge";v="101"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Windows"' \
  --data-raw '{"exclude":["age"],"ids":[[{"tissuename":"GENIE-UHN-OCT111804-ARC1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-UHN-OCT429859-ARC1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-UHN-OCT633473-ARC1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-UHN-OCT695843-ARC1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-UHN-OCT752936-ARC1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-UHN-OCT832227-ARC1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-VHIO-044-001","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-VHIO-177-001","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-VHIO-588-001","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-VICC-194381-unk-1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-VICC-527265-unk-1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-VICC-779846-unk-1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-VICC-912814-unk-1","Cohort":"#2 Gender: Female"},{"tissuename":"GENIE-WAKE-F1606-01","Cohort":"#2 Gender: Female"}],[{"tissuename":"GENIE-MSK-P-0039378-T02-IM6","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-MSK-P-0044286-T01-IM6","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-MSK-P-0047312-T01-IM6","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-MSK-P-0047449-T01-IM6","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-MSK-P-0049749-T01-IM6","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-NKI-QMQ2-XL8Y","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-UCHI-Patient240-T1","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-UCSF-10652-2883T","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-UCSF-1889-2063T","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-UCSF-2291-4904T","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-UCSF-3427-3316T","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-UHN-013895-ARC1","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-UHN-074606-ARC1","Cohort":"#3 Gender: Male"},{"tissuename":"GENIE-UHN-AGI490135-BM1","Cohort":"#3 Gender: Male"}]]}' \
  --compressed