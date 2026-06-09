export interface DemoPlayer {
  jersey_number: number
  name: string
}

export interface DemoTeam {
  id: string
  name: string
  shortName: string
  region: 'Bandung' | 'Surabaya'
  players: DemoPlayer[]
}

export const DEMO_TEAMS: DemoTeam[] = [
  {
    id: 'maranatha',
    name: 'Universitas Kristen Maranatha Bandung',
    shortName: 'Maranatha',
    region: 'Bandung',
    players: [
      { jersey_number: 0,  name: 'Kevin Kristian Simanjuntak' },
      { jersey_number: 1,  name: 'Farrergie Alexandre Nauval' },
      { jersey_number: 2,  name: 'Marcell Syahlevi Putera' },
      { jersey_number: 3,  name: 'Muhamad Rasya Putra Pranata' },
      { jersey_number: 4,  name: 'Likemo Victor Deo Putra Conrad' },
      { jersey_number: 5,  name: 'Aditya Wiguna Sulistiyo' },
      { jersey_number: 7,  name: 'Ivan Matthew Baliputera' },
      { jersey_number: 8,  name: 'Aloysius Valentino Sumana' },
      { jersey_number: 9,  name: 'Ferry Mingtoro' },
      { jersey_number: 10, name: 'Joshua Rafael Kurniawan Lawento' },
      { jersey_number: 11, name: 'Jonathan Prasetyo Rudiyanto' },
      { jersey_number: 15, name: 'I Made Duta Wahyundara' },
      { jersey_number: 17, name: 'Zihad Visabililah' },
      { jersey_number: 21, name: 'Joshua Nathanael Yangtjik' },
      { jersey_number: 22, name: 'Ardino Argya Natareksa' },
      { jersey_number: 27, name: 'Fikran Fattah' },
      { jersey_number: 35, name: 'Jonathan Karsten Hatorangan Panggabean' },
      { jersey_number: 78, name: 'Karel Nazwa Megantara' },
    ],
  },
  {
    id: 'upi',
    name: 'Universitas Pendidikan Indonesia Bandung',
    shortName: 'UPI Bandung',
    region: 'Bandung',
    players: [
      { jersey_number: 0,  name: 'Reysya Reskia Putra' },
      { jersey_number: 4,  name: 'Muhammad Fathurrahman' },
      { jersey_number: 5,  name: 'Fasya Alifa Rifdan' },
      { jersey_number: 6,  name: 'Muhammad Yusuf Abdillah' },
      { jersey_number: 8,  name: 'Gilar Rizky Firdaus' },
      { jersey_number: 9,  name: 'Muhammad Allen Al-Azizu' },
      { jersey_number: 12, name: 'Nizwar Adriansyah Purnama Suparma' },
      { jersey_number: 13, name: 'Kaka Zakki Fadlu Rohim' },
      { jersey_number: 15, name: 'Adrian Rafael Tagas' },
      { jersey_number: 21, name: 'Muhammad Raihan Zuhruful Ulum' },
      { jersey_number: 24, name: 'Raden Mas Fiqih Nawaal Fadilah' },
      { jersey_number: 29, name: 'Naufal Ramdhan Nasrulloh' },
      { jersey_number: 36, name: 'Devin Andhika' },
      { jersey_number: 39, name: 'Muhammad Ridho Muzakir' },
      { jersey_number: 67, name: 'Muhammad Fatih Al Fath' },
      { jersey_number: 73, name: 'Perwira Nabil Zuhdi' },
    ],
  },
  {
    id: 'unesa',
    name: 'Universitas Negeri Surabaya',
    shortName: 'UNESA',
    region: 'Surabaya',
    players: [
      { jersey_number: 4,  name: 'Rio Julian Ramadhani' },
      { jersey_number: 5,  name: 'Aditya Firmansah' },
      { jersey_number: 6,  name: 'Mauludiar Fathir Athalla' },
      { jersey_number: 7,  name: 'Daffa Athalla Defrianto' },
      { jersey_number: 8,  name: 'Syahmi Rianta Aswayni' },
      { jersey_number: 9,  name: 'Akiko Hafiz Maulana' },
      { jersey_number: 10, name: 'Andrew Garcia Christianto' },
      { jersey_number: 11, name: 'Rico Romana' },
      { jersey_number: 12, name: 'Freya Deva Pinastika Manggala' },
      { jersey_number: 13, name: 'Andhika Dwi Iriansyah' },
      { jersey_number: 14, name: 'Damar Widi Sasongko' },
      { jersey_number: 16, name: 'Reinja Caesar Paluin' },
      { jersey_number: 18, name: 'I Nyoman Bagus Bhaskara Daneswara' },
    ],
  },
  {
    id: 'ubaya',
    name: 'Universitas Surabaya',
    shortName: 'UBAYA',
    region: 'Surabaya',
    players: [
      { jersey_number: 0,  name: 'Axel Jevan Whyte' },
      { jersey_number: 1,  name: 'Haviere Noah Richel Kester' },
      { jersey_number: 2,  name: 'Revan Surya Winatha' },
      { jersey_number: 3,  name: 'Benjamin Archie Prasada' },
      { jersey_number: 4,  name: 'Felix Fie Navarine' },
      { jersey_number: 5,  name: 'I Gusti Bagus Hariraja Kinandhana' },
      { jersey_number: 7,  name: 'Johan Son Prasetyo' },
      { jersey_number: 8,  name: 'Dylan Wilbert Pranoto' },
      { jersey_number: 9,  name: 'Macxquel Rehan Thiemailattu' },
      { jersey_number: 10, name: 'Jonathan Patrick Alex' },
      { jersey_number: 11, name: 'Albert Richard' },
      { jersey_number: 12, name: 'Rivaldo Yauwry' },
      { jersey_number: 15, name: 'Jansen Cahyadi' },
      { jersey_number: 17, name: 'Geoffrey Grant Gavindra' },
      { jersey_number: 18, name: 'Steven Arya Giovanni' },
      { jersey_number: 20, name: 'Vitobratta Rohman' },
      { jersey_number: 23, name: 'Bobby Geraldo Alanda' },
      { jersey_number: 32, name: 'Anthonio Andreas Randalembang' },
      { jersey_number: 72, name: 'Johanson Ronald Simbolon' },
      { jersey_number: 88, name: 'Petra Karunia Mariw K. Sinulingga' },
    ],
  },
]
