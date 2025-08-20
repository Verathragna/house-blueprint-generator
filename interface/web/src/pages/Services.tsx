import React from 'react';
import { Ruler, Cuboid as Cube, PenTool, Users, Palette, Clock } from 'lucide-react';

const Services = () => {
  const services = [
    {
      icon: Ruler,
      title: 'Custom Blueprint Design',
      description: 'Professional custom blueprint designs tailored to your specific needs and preferences.',
      features: ['Detailed floor plans', 'Elevation drawings', 'Structural specifications', 'Material recommendations']
    },
    {
      icon: Cube,
      title: '3D Visualization',
      description: 'Bring your plans to life with photorealistic 3D renderings of your future home.',
      features: ['Interior visualization', 'Exterior renders', 'Virtual walkthrough', 'Lighting simulation']
    },
    {
      icon: PenTool,
      title: 'Renovation Planning',
      description: 'Comprehensive renovation planning services to transform your existing space.',
      features: ['Space optimization', 'Cost estimation', 'Timeline planning', 'Permit assistance']
    },
    {
      icon: Users,
      title: 'Expert Consultation',
      description: 'One-on-one consultation with our experienced architects and designers.',
      features: ['Design guidance', 'Technical advice', 'Budget planning', 'Project management']
    },
    {
      icon: Palette,
      title: 'Interior Design',
      description: 'Complete interior design services to perfect every room in your home.',
      features: ['Color schemes', 'Furniture layout', 'Material selection', 'Decor planning']
    },
    {
      icon: Clock,
      title: 'Project Timeline',
      description: 'Efficient project management to keep your design project on schedule.',
      features: ['Milestone planning', 'Progress tracking', 'Regular updates', 'Deadline management']
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <div className="bg-blue-600 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-6 animate-card">Our Services</h1>
          <p className="text-xl max-w-2xl animate-card animation-delay-100">
            Comprehensive blueprint design services to bring your dream home to life, from initial concept to final details.
          </p>
        </div>
      </div>

      {/* Services Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => (
            <div 
              key={index} 
              className={`bg-white rounded-lg shadow-lg p-8 hover:shadow-xl transition-shadow animate-card`}
              style={{ animationDelay: `${(index + 1) * 100}ms` }}
            >
              <service.icon className="h-12 w-12 text-blue-600 mb-6" />
              <h3 className="text-2xl font-semibold mb-4">{service.title}</h3>
              <p className="text-gray-600 mb-6">{service.description}</p>
              <ul className="space-y-2">
                {service.features.map((feature, idx) => (
                  <li key={idx} className="flex items-center text-gray-700">
                    <div className="h-1.5 w-1.5 bg-blue-600 rounded-full mr-2" />
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-gray-50 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-8 animate-card">Ready to Start Your Project?</h2>
          <button className="bg-blue-600 text-white px-8 py-4 rounded-md text-lg font-medium hover:bg-blue-700 transition-colors animate-card animation-delay-100">
            Schedule a Consultation
          </button>
        </div>
      </div>
    </div>
  );
};

export default Services;