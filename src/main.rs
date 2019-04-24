use std::fs::File;
use std::io::{BufRead, BufReader};

use ndarray::prelude::*;
use ndarray::{Array1, Array2};

mod kabsch;
use kabsch::{kabsch_rmsd, rmsd};

#[macro_use]
extern crate clap;
use clap::{Arg, App};


#[derive(Debug)]
struct Atom {
    symbol: String,
    x: f64,
    y: f64,
    z: f64
}

impl Atom {
    // associated function - used to construct atom
    fn read_atom(line: String) -> Atom {

        let line: Vec<&str> = line.trim().split_whitespace().collect();

        let atom = Atom{symbol: line[0].to_string(),
            x: line[1].parse::<f64>().unwrap(),
            y: line[2].parse::<f64>().unwrap(),
            z: line[3].parse::<f64>().unwrap()
        };

        return atom;
    }
}

fn read_xyz(xyz_file: &str) -> (usize, String, Array2<f64>) {

    let xyz_content = File::open(xyz_file).expect("can't read file");
    let mut reader = BufReader::new(xyz_content);

    // read number atoms.
    let mut num_atoms = String::new();
    let _ = reader.read_line(&mut num_atoms);
    let num_atoms = num_atoms.trim().parse::<usize>()
        .expect("can't read first line - expected number of atoms");

    // read header - often filename
    let mut header = String::new();
    let _ = reader.read_line(&mut header)
        .expect("Cannot read header - expected string");
    header = header.trim().to_string();

    // Create 2D matrix for coordinates:
    let mut matrix = Array2::<f64>::zeros((num_atoms,3));

    for (i, line) in reader.lines().enumerate() {
        let atom = Atom::read_atom(line.unwrap());

        matrix[[i,0]] = atom.x;
        matrix[[i,1]] = atom.y;
        matrix[[i,2]] = atom.z;
    }

    return (num_atoms, header, matrix);
}

fn centroid(mat: &Array2<f64>) -> Array1<f64>{
    let center = mat.mean_axis(Axis(0));
    return center;
}

fn main() {

    let matches = App::new("RS RMSD")
                        .version(crate_version!())
                        .author(crate_authors!())
                        .about("Calculates the RMSD value between two .xyz files")
                        .arg(Arg::with_name("file1") 
                            .help("xyz file1")
                            .required(true)
                            .index(1))
                        .arg(Arg::with_name("file2")
                            .help("xyz file2")
                            .required(true)
                            .index(2))
                        .arg(Arg::with_name("kabsch-rmsd")
                            .help("Computes RMSD rotating file1 onto file2")
                            .short("k")
                            .long("kabsch"))
                        .get_matches();
                                

    let filename1 = matches.value_of("file1").unwrap_or("strange file name");
    let filename2 = matches.value_of("file2").unwrap_or("strange file name");

    let (_num_atoms1, _header1, mut coords1) = read_xyz(&filename1);
    let (_num_atoms2, _header2, mut coords2) = read_xyz(&filename2);

    
    if matches.is_present("kabsch-rmsd") {
        let c_coords1 = centroid(&coords1);
        let c_coords2 = centroid(&coords2);
        coords1 = coords1 - c_coords1;
        coords2 = coords2 - c_coords2;

        let rmsd = kabsch_rmsd(&coords1, &coords2);
        println!("kabsch rmsd: {}", rmsd);
    
    } else {
        let rmsd = rmsd(&coords1, &coords2);
        println!("normal rmsd: {}", rmsd);
    }
}