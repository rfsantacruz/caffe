function [ label, image ] = fromDatum( varargin )
%FROMDATUM decode image and label from caffe protobuf.

CHECK(nargin > 0, ['usage: '...
    '[ label, image ] = fromDatum( datum )']); 
datum = varargin{1};

[label, image] = caffe_('from_datum', datum);

end
