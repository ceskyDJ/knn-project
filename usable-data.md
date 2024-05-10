- seznamzpravy
- garaz
- idnes
    - keep top half
- lidovky
    - keep top half
- lupa
    - could cut top half in a lot of screens
- novinky
- sport
- zive


some fixes:
- aktualitysk
    - doesn't have date
    - some are super long, need to be split -- it is too long even for sliding window (OCR won't work)
    - a lot of boxes actually seem to not be correct




# 6. 5.

aha
- [post] potřeba uříznout spodní část stránky, nejsou tam komentáře, je to pak moc vysoké
- je to lehce posunuté, nemuselo by to úplně vadit
    - [ ] šlo by to prostě plošně offsetnout zpátky?
    - zdá se to konzistentně posunuté o jeden znak doprava

aktualitysk
- nefunguje vůbec
- rozbitá stránka celá + cookies

auto
- potřeba uříznout spodní část stránky
- posunuté stejně jako aha

avmania
- nenašlo to žádné komentáře -- i když na screenshotu jsou, nejsou žádné bbox data
- takže bychom mohli použít jako evaluační data

blesk
- potřeba uříznout spodní část stránky
- posunuté o jeden znak doprava

connect
- potřeba uříznout spodní část stránky
- posunuté o jeden znak doprava
- datum bbox se zdá zahrnovat i jiné věci -- author name

doupe
- potřeba uříznout spodní část stránky
- posunuté o jeden znak doprava
- datum bbox se zdá zahrnovat i jiné věci -- author name

e15
- některé stránky jsou hrozně dlouhé, potřeba nakrájet
- posunuto o jeden znak doprava
- text a wrapper bboxy by se mohly trochu zmenšit
- wrapper obsahuje i child komentáře
- there are no 2.png, still keep only 1.png to be sure

idnes
- hodně dlouhé stránky často
    - říznout všechny v polovině
- nechat pouze 1.png každé stránky, ostatní mají velkou šanci, že budou posunuté bboxy
- text a author a wrapper posunuté cca o znak doprava
- datum je taky posunuté, ale většinou to ještě sedí, takže by nemuselo být potřeba to posouvat
- levá strana wrapperu je úplně vlevo, i pro zanořené komentáře
    - šlo by upravit podle levé strany textu, tak se posouvá správně se zanořením
    - toto je asi problém toho, jak je strukturované vlastní html, předpokládám
- vyskytují se posunuté bboxy mimo komentáře úplně
    - např. idnes/6/2, idnes/6/3
        - idnes/8/2, idnes/8/3, ...
    - hádám že to má něco společného ss reklamou, jelikož před rozbitými komentáři je volné místo,
      kde asi měla být reklama
    - taky se zdá, že se to děje na "dalších stránkách". často vidím, že 1.png je v pořádku, ale pokud to má 2.png, 3.png, ...
      tak nastává problém
- odpovědi od iDNES týmu se neregistrují jako komentáře -- idnes/72/1 (Pavel Kočička)
- na 84/1, se neregistrují odpovědi
- 612/1, 828/1, 832/1 - jedna odpověď není registrována
- na 87/1 se neregistroval jeden komentář
- filter-out:
    - 12,72,84,87,283,605,612,828 832,1050,1328,1335,1620,1623,1631,1957,2316,3104,4041,4053,4543,5089,6244

isport
- stánky hrozně dlouhé -- uříznout v půlce minimálně
- skoro žádné komentáře
- posunuté o znak doprava
- filter-out
    - 12

lidovky
- uříznout v půlce
- posunuto a znak doprava

lupa
- uříznout většinu článku v horní polovině
- posunuté o znak doprava
- posunuté o cca komentář dolů
- můžeme zkusit použít pro experiment jak dobře to generalizuje

mobilmania
- posunuté o znak doprava
- datum není správně označeno
- museli bychom uměle vytvořit bbox pro datum -- začíná na stejném x jako autor, o "řádek" pod autorem

pravda
- posunuté o znak doprava
- není správně označené datum vůbec

sme
- určitě potřeba nakrájet -- hrozně dlouhé
- posunuté o znak doprava
- *někdy* posunuté o cca 2 řádky dolů -- offset vypadá konstantní, mohlo by jít opravit
    - možná teda lepší detekovat a úplně ignoroavt
- někdy text zahrnuje parent reference, někdy ne, ale tak jako tak tam není bbox pro parent reference
    - spíš to teda jako parent reference označuje náhodné odkazy vrámci komentáře (sme/1686/1)
- filter-out
    - 1
    - TODO?

vtm
- stejné jako mobilmania

zive
- stejně jako mobilmania
