/**
 * @fileoverview About page component that showcases company information,
 * team members, and key statistics.
 */

import React from 'react';

/**
 * @typedef {Object} StatItem
 * @property {React.ComponentType} icon - Lucide icon component
 * @property {string} value - Display value for the statistic
 * @property {string} label - Description of the statistic
 */

/**
 * @typedef {Object} TeamMember
 * @property {string} name - Team member's full name
 * @property {string} role - Professional role/title
 * @property {string} image - URL to profile image (from Unsplash)
 * @property {string} bio - Brief professional biography
 */

const About = () => {
  // Company statistics displayed in the stats section
  const stats = [
    { icon: Award, value: '15+', label: 'Years Experience' },
    { icon: Users, value: '500+', label: 'Happy Clients' },
    { icon: Clock, value: '1000+', label: 'Projects Completed' },
    { icon: CheckCircle, value: '100%', label: 'Satisfaction Rate' }
  ];

  // Team member information with profile images from Unsplash
  const team = [
    {
      name: 'John Anderson',
      role: 'Principal Architect',
      image: 'https://images.unsplash.com/photo-1600180758890-6b94519a8ba6?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80',
      bio: '20 years of experience in residential architecture.'
    },
    {
      name: 'Sarah Martinez',
      role: '3D Visualization Specialist',
      image: 'https://images.unsplash.com/photo-1600180759207-fa9bd62dd2d7?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80',
      bio: 'Expert in creating photorealistic 3D renderings.'
    },
    {
      name: 'David Kim',
      role: 'Interior Designer',
      image: 'https://images.unsplash.com/photo-1600180758974-c2c5b3faeb0a?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80',
      bio: 'Specializes in modern and contemporary design.'
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section with parallax background effect */}
      <div 
        className="relative h-[60vh] bg-cover bg-center"
        style={{
          backgroundImage: 'url("https://images.unsplash.com/photo-1600607687920-4e2a09cf159d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80")'
        }}
      >
        <div className="absolute inset-0 bg-black/50" />
        <div className="relative h-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center">
          <div className="text-white">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">About BlueprintPro</h1>
            <p className="text-xl max-w-2xl">
              We're a team of passionate architects and designers dedicated to creating exceptional living spaces that inspire and endure.
            </p>
          </div>
        </div>
      </div>

      {/* Stats Section - Key company metrics */}
      <div className="bg-blue-600 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <stat.icon className="h-8 w-8 mx-auto mb-4" />
                <div className="text-3xl font-bold mb-2">{stat.value}</div>
                <div className="text-blue-100">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Mission Statement Section */}
      <div className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
            <p className="text-xl text-gray-600">
              To transform living spaces through innovative design, creating homes that perfectly balance aesthetics, functionality, and sustainability while exceeding our clients' expectations.
            </p>
          </div>
        </div>
      </div>

      {/* Team Section - Profile cards for key team members */}
      <div className="bg-gray-50 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">Meet Our Team</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {team.map((member, index) => (
              <div key={index} className="bg-white rounded-lg shadow-lg overflow-hidden">
                <img
                  src={member.image}
                  alt={member.name}
                  className="w-full h-64 object-cover"
                />
                <div className="p-6">
                  <h3 className="text-xl font-semibold mb-2">{member.name}</h3>
                  <p className="text-blue-600 mb-4">{member.role}</p>
                  <p className="text-gray-600">{member.bio}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;