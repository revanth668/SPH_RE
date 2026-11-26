# SPH: From Merging Particles to the Moon-Forming Giant Impact

A week-by-week study of **Smoothed Particle Hydrodynamics (SPH)**, a mesh-free method in which a fluid or body is represented by particles whose properties are smoothed by a kernel. The project starts from the theory, builds 2D collision and orbit simulations in Python, and ends with a browser model of how the Moon may have formed.

**Core physics:** cubic spline kernel, summation density, pressure forces, artificial viscosity, leapfrog integration, Newtonian gravity, and perfectly inelastic merging (mass and momentum conserved, kinetic energy lost as heat).

## Weekly progress

| Week | What was done | Main file(s) |
|---|---|---|
| 1 | Presentation on Monaghan (1992), Sections 1–4 | `SPH_Presentation week-1.pptx` |
| 2 | 2D SPH with 4 particles that merge on collision | `sph_merge_sim (1).py` |
| 3 | 1,000 particles; merge when radii overlap; optional GPU (CuPy) | `sph_merge_sim.py` |
| 4 | Exact contact-time collisions (no SPH forces), final positions, mass-distribution growth. 300 particles → 131, with 169 merges | `SPH_merge_report.docx`, `inelastic_merge_sim.py` |
| 5 | Mutual gravity added; GIFs compare attraction and repulsion over different mass ranges | `sph_merge_sim_2.py` |
| 6 | Sun and two planets collide and merge into a new orbit (toy units) | `sph_orbit_collision_sim.py` |
| 7 | Real-scale Sun–Earth–Theia: impact on day 218 at 9.8 km/s, stable merged orbit. Also a reference giant-impact movie (`119_1m.avi`) | `sph_orbit_collision_real_scale.py` + `_EXPLANATION.md` |
| 8 | Impactor flies past Mars (day 110), shortening Mars's year by 64 days, then hits Earth (day 231, 11.6 km/s) | `sph_mars_flyby_impact.py` + `_EXPLANATION.md` |
| 9 | Browser model after Canup (2012): impact → debris disk → Moon, with many particles per planet | `moon-impact.html`, `sim.js` |
| 10 | Refined model: temperature, faster neighbour search, up to 8,000 particles, cooling toggle | `moon-impact (2).html`, `sim (1).js` |

## How to run

**Python (Weeks 2–8):** install the libraries, then run a script from inside its week folder, since outputs are saved to the current folder.

```bash
pip install numpy matplotlib pillow scipy
cd "Week 8"
python sph_mars_flyby_impact.py
```

- SciPy is required for Week 8 and recommended for Week 7. Install ffmpeg for MP4 output; without it, Weeks 7–8 save GIFs.
- CuPy is optional and only worthwhile for large particle counts.
- The Week 5 script holds one setting (15 particles, mass 10–11, attractive). Edit the mass range or `G` to try the other cases.

**Moon impact (Weeks 9–10):** open the HTML file in a browser, choose a scenario (Canup or Mars-sized impactor), and click **Run impact**. No installation needed.

## Limitations

- All simulations are 2D.
- Weeks 6–8 use **one particle per body**, so they model orbits and merging, not interiors, debris or Moon formation.
- Weeks 9–10 are toy models with simplified equations of state and far fewer particles than Canup's 300,000. Expect the right sequence of events, not the paper's numbers.

## References

1. J. J. Monaghan, "Smoothed Particle Hydrodynamics," *Annu. Rev. Astron. Astrophys.* **30**, 543–574 (1992): `annurev.aa.30.090192.002551 (1).pdf`
2. R. M. Canup, "Forming a Moon with an Earth-like Composition via a Giant Impact," *Science* **338**, 1052 (2012): `Week 9/canup2012.pdf`
