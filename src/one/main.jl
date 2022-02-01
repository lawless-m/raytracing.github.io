
using StaticArrays
using LinearAlgebra


const Vec3 = SVector{3}
const Point3 = SVector{3}

magnitude(v::Vec3) = sqrt(v.x^2 + v.y^2 + v.z^2)
unit_vector(v::Vec3) = v / magnitude(v)

near_zero(v) = v.x < 1e-8 && v.y < 1e-8 && v.z < 1e-8


function random_in_unit_disk() 
	newv() = Vec3(randf(-1, 1), randf(-1, 1), 0)
	v = newv()
	while magnitude(v)^2 >= 1
        	v = newv()
	end
        v
end

function random_in_unit_sphere() 
	newv() = Vec3(randf(-1,1), randf(-1,1), randf(-1,1))
	v = newv()
	while magnitude(v)^2 >= 1
		v = newv()
	end
	v
end

random_unit_vector() = unit_vector(random_in_unit_sphere())

function random_in_hemisphere(normal) 
    in_unit_sphere = random_in_unit_sphere()
    dot(in_unit_sphere, normal) > 0.0 ? in_unit_sphere : -in_unit_sphere
end

reflect(v, n) =  v - 2dot(v,n)*n

function refract(uv, n, etai_over_etat) 
    cos_theta = min(dot(-uv, n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta*n)
    r_out_parallel = -sqrt(abs(1.0 - magnitude(r_out_perp)^2)) * n
    r_out_perp + r_out_parallel
end

struct Ray
	origin::Point3
	direction::Vec3
	tm::Float64
	Ray(o, d, m) = new(o, d, m)
	Ray(o, d) = Ray(o, d, 0)
end


at(r::Ray, t) = r.origin + t * r.direction

struct Camera
        origin::Point3
        lower_left_corner::Point3
        horizontal::Vec3
        vertical::Vec3
	u::Vec3
	v::Vec3
	w::Vec3
        lens_radius::Float64
	time0::Float64
	time1::Float64
	function Camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist, _time0=0, _time1=0)
            theta = deg2rad(vfov)
            h = tan(theta/2)
            viewport_height = 2.0 * h
            viewport_width = aspect_ratio * viewport_height

            w = unit_vector(lookfrom - lookat)
            u = unit_vector(cross(vup, w))
            v = cross(w, u)

            origin = lookfrom
            horizontal = focus_dist * viewport_width * u
            vertical = focus_dist * viewport_height * v
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w

            lens_radius = aperture / 2
            time0 = _time0
            time1 = _time1
		new(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius, time0, time1)
	end
	Camera() = Camera(Point3(0,0,-1), Point3(0,0,0), Vec3(0,1,0), 40, 1, 0, 10)
end

abstract type Material end
abstract type Hitable end

struct Scene
	camera::Camera
	hitables::Vector{Hitable}
	Scene(cam) = new(cam, Vector{Hitable}())
end

randf(fmin, fmax) = fmin + (fmax-fmin)*rand()

get_ray(scene::Scene, s, t) = get_ray(scene.camera, s, t)

function get_ray(cam::Camera, s, t)
	rd = cam.lens_radius * random_in_unit_disk()
	offset = cam.u * rd.x + cam.v * rd.y
	Ray(cam.origin + offset, cam.lower_left_corner + s*cam.horizontal + t*cam.vertical - cam.origin - offset, randf(cam.time0, cam.time1))
end

struct Color
	rgb::Point3
	Color(r,g,b) = new(Point3(r,g,b))
	Color(n) = Color(n,n,n)
	Color() = Color(rand(), rand(), rand())
end

import Base.+
+(c1::Color, c2::Color) = Color(c1.rgb[1]+c2.rgb[1], c1.rgb[2]+c2.rgb[2], c1.rgb[3]+c2.rgb[3])

import Base.*
*(f::Float64, c::Color) = Color(f*c.rgb[1], f+c.rgb[2], f*c.rgb[3])
*(c1::Color, c2::Color) = Color(c1.rgb[1]*c2.rgb[1], c1.rgb[2]*c2.rgb[2], c1.rgb[3]*c2.rgb[3])

import Base./
/(c::Color, f) = Color(c.rgb[1]/f, c.rgb[2]/f, c.rgb[3]/f)


clamp(f, b, t) = f < b ? b : f > t ? t : f

function write(io::IO, c::Color)
	val(rgb) = round(Int, 256clamp(sqrt(rgb), 0, 1))
	println(io, "$(val(c.rgb[1])) $(val(c.rgb[2])) $(val(c.rgb[3]))")
end


struct Hit 
	p::Point3
	normal::Vec3
	material::Material
	t::Float64
	front_face::Bool
	function Hit(p, t, ray, outward_normal, material) # sphere
		ff = dot(ray.direction, outward_normal)
		norm = ff < 0 ? outward_normal : -outward_normal
		new(p, norm, material, t, ff < 0)
	end
end

function trace(scene::Scene, ray::Ray, t_min, t_max)
	closest_t = t_max
	nearest = nothing
	
	for hitable in scene.hitables
		hit = trace(hitable, ray, t_min, closest_t)
		if hit !== nothing
			closest_t = hit.t
			nearest = hit
		end
	end
	nearest
end


struct Lambertian <: Material
	albedo::Color
	Lambertian(c::Color) = new(c)
	Lambertian(r,g,b) = new(Color(r,g,b))
	Lambertian() = Lambertian(Color(rand()*rand(), rand()*rand(), rand()*rand()))
end

function scatter(l::Lambertian, ray::Ray, hit::Hit)
	scatter_direction = hit.normal + random_unit_vector()
	if near_zero(scatter_direction)
		scatter_direction = hit.normal
	end
	Ray(hit.p, scatter_direction), l.albedo
end

struct Metal <: Material
	albedo::Color
	fuzz::Float64
	Metal(a, f) = new(a, f)
	Metal(r,g,b,f) = Metal(Color(r,g,b), f)
end

function scatter(m::Metal, ray::Ray, hit::Hit)
	reflected = reflect(unit_vector(ray.direction), hit.normal)
        scattered = Ray(hit.p, reflected + m.fuzz * random_in_unit_sphere())
	dot(scattered.direction, hit.normal) > 0 ? (scattered, m.albedo) : nothing
end

struct Dielectric <: Material
	ir::Float64
end

function reflectance(cosine, ref_idx)
	r0 = ((1-ref_idx) / (1+ref_idx))^2
        r0 + (1-r0)*(1 - cosine)^5
end

function scatter(d::Dielectric, ray::Ray, hit::Hit) 
	refraction_ratio = hit.front_face ? (1.0/d.ir) : d.ir

	unit_direction = unit_vector(ray.direction)
	cos_theta = min(dot(-unit_direction, hit.normal), 1.0)
	sin_theta = sqrt(1.0 - cos_theta^2)
	cannot_refract = refraction_ratio * sin_theta > 1.0
	direction = if cannot_refract || reflectance(cos_theta, refraction_ratio) > rand()
			reflect(unit_direction, hit.normal)
		else
			refract(unit_direction, hit.normal, refraction_ratio)
		end

	Ray(hit.p, direction), Color(1)
end

abstract type Hitable end

struct Sphere <: Hitable
	center::Point3
	radius::Float64
	material::Material
end


function trace(sphere::Sphere, ray::Ray, t_min, t_max)
	oc = ray.origin - sphere.center
	a = magnitude(ray.direction)^2
	half_b = dot(oc, ray.direction)
	c = magnitude(oc)^2 - sphere.radius^2
	discriminant = half_b^2 - a*c
	if discriminant < 0
		return
	end

	sqrtd = sqrt(discriminant)
	root = (-half_b - sqrtd) / a
	if root < t_min || t_max < root
		root = (-half_b + sqrtd) / a
		if root < t_min || t_max < root
			return 
		end
	end

	p = at(ray, root)

	Hit(p, root, ray, (p - sphere.center) / sphere.radius, sphere.material)
end


function ray_color(scene::Scene, ray::Ray, depth) 
	if depth <= 0
        	return zero(Color)
	end

	hit = trace(scene, ray, 0.001, Inf)
	if hit !== nothing
		sa = scatter(hit.material, ray, hit)
		if sa === nothing
			return zero(Color)
		end
		return sa[2] * ray_color(scene, sa[1], depth-1)
	end

	unit_direction = unit_vector(ray.direction)
	t = 0.5*(unit_direction.y + 1.0)
	(1.0-t)*Color(1) + t*Color(0.5, 0.7, 1.0)
end

add!(s::Scene, h::Hitable) = push!(s.hitables, h)


function add_random_scene!(scene::Scene) 

	add!(scene, Sphere(Point3(0,-1000,0), 1000, Lambertian(Color(0.5))))

	rand_material(p) = if p < 0.8 
				Lambertian()
			elseif p < 0.95
				Metal(Color(randf(0.5, 1)), 0.5rand())
			else
				Dielectric(1.5)
			end

	for a in -11:10, b in -11:10
		center = Point3(a + 0.9rand(), 0.2, b + 0.9rand())
		if magnitude(center - Point3(4, 0.2, 0)) > 0.9
			add!(scene, Sphere(center, 0.2, rand_material(rand())))
		end
	end

	add!(scene, Sphere(Point3(0, 1, 0), 1.0, Dielectric(1.5)))
	add!(scene, Sphere(Point3(-4, 1, 0), 1.0, Lambertian(0.4,0.2,0.1)))
	add!(scene, Sphere(Point3(4, 1, 0), 1.0, Metal(Color(0.7,0.6,0.5), 0.0)))
end

zero(::Type{Point3}) = Point3(0,0,0)
zero(::Type{Color}) = Color(0)


function main(io, image_width=1200, aspect_ratio=16.0/9.0)

	image_height = round(Int, image_width / aspect_ratio)
	samples_per_pixel = 10
	max_depth = 50

    	lookfrom = Point3(13.,2.,3.)
	lookat = zero(Point3)


	vup = Vec3(0,1,0)
	dist_to_focus = 10.0
	aperture = 0.1

	world = Scene(Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus))
	add_random_scene!(world)

	println(io, "P3\n$image_width $image_height\n255")

	for j in image_height-1:-1:0

		print(stderr, "\rScanlines remaining: $j   ")
		flush(stderr)

		for i in 0:image_width-1
            		pixel_color = zero(Color)
			for _ in 1:samples_per_pixel
                		u = (i + rand()) / (image_width-1)
                		v = (j + rand()) / (image_height-1)
                		r = get_ray(world, u, v)
                		pixel_color += ray_color(world, r, max_depth)
			end
            		write(io, pixel_color / samples_per_pixel)
  		end
       end

	println(stderr, "Done")
end



