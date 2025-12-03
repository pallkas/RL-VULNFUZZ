#!/usr/bin/env python3
"""
SPEC CPU2006 Seed Generator for AFL++ Fuzzing

Creates minimal valid seeds for each SPEC benchmark that will
execute quickly (avoid timeout) while still exercising code paths.
"""

import os
import struct
import random

def create_seed_dir(base_dir="./spec_seeds"):
    """Create organized seed directories"""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def generate_bzip2_seeds(output_dir):
    """
    bzip2 - compression/decompression
    Needs: Small files to compress, valid bz2 files to decompress
    """
    seed_dir = os.path.join(output_dir, "bzip2")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Tiny files that compress almost instantly
    with open(os.path.join(seed_dir, "tiny_1byte"), 'wb') as f:
        f.write(b'A')
    
    with open(os.path.join(seed_dir, "tiny_4byte"), 'wb') as f:
        f.write(b'AAAA')
    
    with open(os.path.join(seed_dir, "tiny_16byte"), 'wb') as f:
        f.write(b'A' * 16)
    
    with open(os.path.join(seed_dir, "zeros_8byte"), 'wb') as f:
        f.write(b'\x00' * 8)
    
    with open(os.path.join(seed_dir, "mixed_16byte"), 'wb') as f:
        f.write(b'\x00\x01\x02\x03' * 4)
    
    # Minimal valid bz2 header (for decompression paths)
    # BZ header + minimal stream
    bz2_minimal = bytes([
        0x42, 0x5A,  # 'BZ' magic
        0x68,        # 'h' for bzip2
        0x39,        # Block size (900k)
        0x31, 0x41, 0x59, 0x26, 0x53, 0x59,  # Pi hex digits (block magic)
    ])
    with open(os.path.join(seed_dir, "bz2_header"), 'wb') as f:
        f.write(bz2_minimal)
    
    print(f"  Created {len(os.listdir(seed_dir))} seeds for bzip2")
    return seed_dir

def generate_gcc_seeds(output_dir):
    """
    gcc - C compiler
    Needs: Minimal C source files
    """
    seed_dir = os.path.join(output_dir, "gcc")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Minimal C programs
    seeds = [
        ("empty.c", b""),
        ("minimal.c", b"int main(){}"),
        ("return.c", b"int main(){return 0;}"),
        ("var.c", b"int main(){int x=1;return x;}"),
        ("if.c", b"int main(){if(1)return 0;return 1;}"),
        ("loop.c", b"int main(){for(int i=0;i<1;i++);return 0;}"),
        ("func.c", b"void f(){}int main(){f();return 0;}"),
        ("ptr.c", b"int main(){int *p=0;return 0;}"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    print(f"  Created {len(seeds)} seeds for gcc")
    return seed_dir

def generate_mcf_seeds(output_dir):
    """
    mcf - Minimum cost flow (network optimization)
    Needs: Network flow problem input format
    """
    seed_dir = os.path.join(output_dir, "mcf")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Minimal network problem format
    # Usually: nodes, arcs, supply/demand
    seeds = [
        ("tiny.net", b"1 0\n"),  # 1 node, 0 arcs
        ("small.net", b"2 1\n1 2 1 1 0\n"),  # 2 nodes, 1 arc
        ("nums.net", b"0 0\n"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    # Also generic small inputs
    for i in range(5):
        with open(os.path.join(seed_dir, f"rand_{i}"), 'wb') as f:
            f.write(os.urandom(random.randint(4, 32)))
    
    print(f"  Created {len(os.listdir(seed_dir))} seeds for mcf")
    return seed_dir

def generate_gobmk_seeds(output_dir):
    """
    gobmk - Go game AI
    Needs: GTP commands or SGF game files
    """
    seed_dir = os.path.join(output_dir, "gobmk")
    os.makedirs(seed_dir, exist_ok=True)
    
    # GTP (Go Text Protocol) commands
    seeds = [
        ("quit.gtp", b"quit\n"),
        ("name.gtp", b"name\nquit\n"),
        ("version.gtp", b"version\nquit\n"),
        ("boardsize.gtp", b"boardsize 9\nquit\n"),
        ("empty.gtp", b"\n"),
        ("play.gtp", b"play black A1\nquit\n"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    # Minimal SGF (Smart Game Format)
    sgf = b"(;GM[1]SZ[9])"
    with open(os.path.join(seed_dir, "mini.sgf"), 'wb') as f:
        f.write(sgf)
    
    print(f"  Created {len(os.listdir(seed_dir))} seeds for gobmk")
    return seed_dir

def generate_hmmer_seeds(output_dir):
    """
    hmmer - Hidden Markov Model for protein sequences
    Needs: FASTA format sequences or HMM files
    """
    seed_dir = os.path.join(output_dir, "hmmer")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Minimal FASTA sequences
    seeds = [
        ("mini.fa", b">seq1\nA\n"),
        ("small.fa", b">seq1\nACGT\n"),
        ("protein.fa", b">prot\nMKL\n"),
        ("multi.fa", b">s1\nA\n>s2\nC\n"),
        ("empty.fa", b">\n\n"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    print(f"  Created {len(seeds)} seeds for hmmer")
    return seed_dir

def generate_sjeng_seeds(output_dir):
    """
    sjeng - Chess AI
    Needs: Chess positions or commands
    """
    seed_dir = os.path.join(output_dir, "sjeng")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Chess commands/moves
    seeds = [
        ("quit.txt", b"quit\n"),
        ("move.txt", b"e2e4\nquit\n"),
        ("fen.txt", b"setboard rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nquit\n"),
        ("analyze.txt", b"analyze\nquit\n"),
        ("new.txt", b"new\nquit\n"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    print(f"  Created {len(seeds)} seeds for sjeng")
    return seed_dir

def generate_h264ref_seeds(output_dir):
    """
    h264ref - H.264 video encoder reference
    Needs: Configuration files or raw video frames
    """
    seed_dir = os.path.join(output_dir, "h264ref")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Minimal config-like files
    seeds = [
        ("empty.cfg", b""),
        ("mini.cfg", b"# config\n"),
        ("param.cfg", b"FramesToBeEncoded = 1\n"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    # Small binary data (could be interpreted as raw frames)
    for i in range(3):
        with open(os.path.join(seed_dir, f"frame_{i}.yuv"), 'wb') as f:
            f.write(os.urandom(64))
    
    print(f"  Created {len(os.listdir(seed_dir))} seeds for h264ref")
    return seed_dir

def generate_libquantum_seeds(output_dir):
    """
    libquantum - Quantum computer simulation
    Needs: Quantum circuit parameters
    """
    seed_dir = os.path.join(output_dir, "libquantum")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Small numeric inputs
    seeds = [
        ("small.txt", b"2\n"),
        ("tiny.txt", b"1\n"),
        ("zero.txt", b"0\n"),
        ("num.txt", b"5\n"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    print(f"  Created {len(seeds)} seeds for libquantum")
    return seed_dir

def generate_milc_seeds(output_dir):
    """
    milc - Lattice QCD simulation
    Needs: Lattice configuration files
    """
    seed_dir = os.path.join(output_dir, "milc")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Minimal config-like inputs
    seeds = [
        ("mini.in", b"prompt 0\n"),
        ("tiny.in", b"1\n"),
        ("empty.in", b"\n"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    print(f"  Created {len(seeds)} seeds for milc")
    return seed_dir

def generate_perlbench_seeds(output_dir):
    """
    perlbench - Perl interpreter
    Needs: Perl scripts
    """
    seed_dir = os.path.join(output_dir, "perlbench")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Minimal Perl scripts
    seeds = [
        ("empty.pl", b""),
        ("exit.pl", b"exit;"),
        ("print.pl", b"print 1;"),
        ("var.pl", b"$x=1;"),
        ("hello.pl", b'print "x";'),
        ("loop.pl", b"for(1..1){}"),
        ("if.pl", b"if(1){}"),
        ("sub.pl", b"sub f{}f();"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    print(f"  Created {len(seeds)} seeds for perlbench")
    return seed_dir

def generate_sphinx_seeds(output_dir):
    """
    sphinx - Speech recognition
    Needs: Audio data or transcripts
    """
    seed_dir = os.path.join(output_dir, "sphinx")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Minimal audio-like data
    # WAV header (minimal)
    wav_header = bytes([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x00, 0x00, 0x00,  # File size - 8
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # Subchunk size
        0x01, 0x00,              # Audio format (PCM)
        0x01, 0x00,              # Num channels (mono)
        0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
        0x88, 0x58, 0x01, 0x00,  # Byte rate
        0x02, 0x00,              # Block align
        0x10, 0x00,              # Bits per sample
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x00, 0x00, 0x00,  # Data size
    ])
    
    with open(os.path.join(seed_dir, "mini.wav"), 'wb') as f:
        f.write(wav_header)
    
    # Raw audio data
    with open(os.path.join(seed_dir, "raw.pcm"), 'wb') as f:
        f.write(b'\x00' * 64)
    
    with open(os.path.join(seed_dir, "silence.raw"), 'wb') as f:
        f.write(b'\x80' * 64)
    
    print(f"  Created {len(os.listdir(seed_dir))} seeds for sphinx")
    return seed_dir

def generate_lbm_seeds(output_dir):
    """
    lbm - Lattice Boltzmann Method (fluid dynamics)
    Needs: Grid/lattice configuration
    """
    seed_dir = os.path.join(output_dir, "lbm")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Small numeric inputs
    for i, size in enumerate([1, 2, 4, 8, 16]):
        with open(os.path.join(seed_dir, f"grid_{i}.dat"), 'wb') as f:
            f.write(struct.pack('f' * size, *[0.0] * size))
    
    # Text config
    with open(os.path.join(seed_dir, "config.txt"), 'wb') as f:
        f.write(b"1 1 1\n")
    
    print(f"  Created {len(os.listdir(seed_dir))} seeds for lbm")
    return seed_dir

def generate_generic_seeds(output_dir):
    """Generate generic seeds that work for many programs"""
    seed_dir = os.path.join(output_dir, "generic")
    os.makedirs(seed_dir, exist_ok=True)
    
    seeds = [
        ("empty", b""),
        ("null", b"\x00"),
        ("one", b"1"),
        ("newline", b"\n"),
        ("space", b" "),
        ("a", b"A"),
        ("hello", b"hello"),
        ("zeros_8", b"\x00" * 8),
        ("ones_8", b"\xff" * 8),
        ("nums", b"12345678"),
    ]
    
    for name, content in seeds:
        with open(os.path.join(seed_dir, name), 'wb') as f:
            f.write(content)
    
    print(f"  Created {len(seeds)} generic seeds")
    return seed_dir

def main():
    print("=" * 60)
    print("SPEC CPU2006 SEED GENERATOR FOR AFL++")
    print("=" * 60)
    print()
    
    base_dir = "./spec_seeds"
    print(f"Creating seeds in: {base_dir}/")
    print()
    
    create_seed_dir(base_dir)
    
    # Generate seeds for each benchmark
    generate_bzip2_seeds(base_dir)
    generate_gcc_seeds(base_dir)
    generate_gobmk_seeds(base_dir)
    generate_h264ref_seeds(base_dir)
    generate_hmmer_seeds(base_dir)
    generate_lbm_seeds(base_dir)
    generate_libquantum_seeds(base_dir)
    generate_mcf_seeds(base_dir)
    generate_milc_seeds(base_dir)
    generate_perlbench_seeds(base_dir)
    generate_sjeng_seeds(base_dir)
    generate_sphinx_seeds(base_dir)
    generate_generic_seeds(base_dir)
    
    print()
    print("=" * 60)
    print("SEED GENERATION COMPLETE!")
    print("=" * 60)
    print()
    print("Usage examples:")
    print()
    print("  # Fuzz bzip2 with bzip2-specific seeds:")
    print(f"  afl-fuzz -i {base_dir}/bzip2 -o ./output -Q -t 10000+ -- ./bzip2_base @@ ")
    print()
    print("  # Fuzz gcc with gcc-specific seeds:")
    print(f"  afl-fuzz -i {base_dir}/gcc -o ./output -Q -t 30000+ -- ./gcc_base @@")
    print()
    print("  # Use generic seeds for any benchmark:")
    print(f"  afl-fuzz -i {base_dir}/generic -o ./output -Q -- ./any_binary @@")
    print()
    print("TIP: Use -t 10000+ or -t 0 for SPEC benchmarks (they're slow!)")
    print("     The '+' suffix means 'skip test cases that timeout'")

if __name__ == "__main__":
    main()
