import React from 'react';

const Portfolio = () => {
  const projects = [
    {
      title: 'Modern Minimalist Home',
      category: 'Residential',
      image: 'https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80',
      description: 'Contemporary design with clean lines and open spaces.',
      details: ['4 Bedrooms', '3 Bathrooms', '2,500 sq ft']
    },
    {
      title: 'Luxury Villa Renovation',
      category: 'Renovation',
      image: 'https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80',
      description: 'Complete renovation of a Mediterranean-style villa.',
      details: ['5 Bedrooms', '4 Bathrooms', '3,800 sq ft']
    },
    {
      title: 'Urban Apartment Complex',
      category: 'Multi-Unit',
      image: 'https://images.unsplash.com/photo-1600607687644-c7171b42498b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80',
      description: 'Modern apartment complex in the heart of the city.',
      details: ['12 Units', 'Rooftop Garden', 'Parking Garage']
    },
    {
      title: 'Eco-Friendly Cottage',
      category: 'Sustainable',
      image: 'https://images.unsplash.com/photo-1600607688969-a5bfcd646154?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80',
      description: 'Sustainable design with minimal environmental impact.',
      details: ['Solar Panels', 'Rainwater Collection', '1,800 sq ft']
    },
    {
      title: 'Traditional Family Home',
      category: 'Residential',
      image: 'https://images.unsplash.com/photo-1600566753376-12c8ab7fb75b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80',
      description: 'Classic design with modern amenities.',
      details: ['5 Bedrooms', '3.5 Bathrooms', '3,200 sq ft']
    },
    {
      title: 'Beach House',
      category: 'Vacation',
      image: 'https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80',
      description: 'Oceanfront property with panoramic views.',
      details: ['3 Bedrooms', '2 Bathrooms', '2,000 sq ft']
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <div className="bg-gray-900 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-6 animate-card">Our Portfolio</h1>
          <p className="text-xl max-w-2xl animate-card animation-delay-100">
            Explore our collection of successful projects and innovative designs that showcase our expertise in residential architecture.
          </p>
        </div>
      </div>

      {/* Portfolio Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <div 
              key={index} 
              className="group relative overflow-hidden rounded-lg shadow-lg animate-card"
              style={{ animationDelay: `${(index + 1) * 100}ms` }}
            >
              <div className="relative h-80">
                <img
                  src={project.image}
                  alt={project.title}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
              </div>
              <div className="absolute bottom-0 left-0 right-0 p-6 text-white">
                <span className="text-sm font-medium bg-blue-600 px-3 py-1 rounded-full">
                  {project.category}
                </span>
                <h3 className="text-xl font-semibold mt-2">{project.title}</h3>
                <p className="text-gray-200 mt-1">{project.description}</p>
                <div className="flex gap-4 mt-3">
                  {project.details.map((detail, idx) => (
                    <span key={idx} className="text-sm bg-black/30 px-3 py-1 rounded-full">
                      {detail}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Portfolio;